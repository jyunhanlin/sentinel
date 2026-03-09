from __future__ import annotations

import structlog
from pydantic import BaseModel, ConfigDict

from orchestrator.agents.base import BaseAgent
from orchestrator.llm.client import LLMCallResult
from orchestrator.models import CritiqueResult, TradeProposal

logger = structlog.get_logger(__name__)


class RefinementResult(BaseModel):
    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    proposal: TradeProposal
    critique: CritiqueResult | None = None
    rounds: int = 0
    exhausted: bool = False
    all_llm_calls: list[LLMCallResult] = []
    proposer_degraded: bool = False
    critic_degraded: bool = False


class RefinementLoop:
    def __init__(
        self,
        *,
        proposer: BaseAgent[TradeProposal],
        critic: BaseAgent[CritiqueResult],
        max_rounds: int = 2,
    ) -> None:
        self._proposer = proposer
        self._critic = critic
        self._max_rounds = max_rounds

    async def run(
        self,
        *,
        model_override: str | None = None,
        **proposer_kwargs,
    ) -> RefinementResult:
        all_llm_calls: list[LLMCallResult] = []

        # Initial proposer call
        proposer_result = await self._proposer.analyze(
            model_override=model_override, **proposer_kwargs,
        )
        all_llm_calls.extend(proposer_result.llm_calls)

        if proposer_result.degraded:
            logger.warning("refinement_proposer_degraded")
            return RefinementResult(
                proposal=proposer_result.output,
                proposer_degraded=True,
                all_llm_calls=all_llm_calls,
            )

        proposal = proposer_result.output
        critique: CritiqueResult | None = None

        for round_num in range(1, self._max_rounds + 1):
            logger.info("refinement_round", round=round_num)

            # Run critic
            critic_result = await self._critic.analyze(
                proposal=proposal,
                model_override=model_override,
                **proposer_kwargs,
            )
            all_llm_calls.extend(critic_result.llm_calls)
            critique = critic_result.output

            if critic_result.degraded:
                logger.warning("refinement_critic_degraded", round=round_num)
                return RefinementResult(
                    proposal=proposal,
                    critique=critique,
                    rounds=round_num,
                    critic_degraded=True,
                    all_llm_calls=all_llm_calls,
                )

            if critique.overall_passed:
                logger.info("refinement_passed", round=round_num)
                return RefinementResult(
                    proposal=proposal,
                    critique=critique,
                    rounds=round_num,
                    all_llm_calls=all_llm_calls,
                )

            # Critique failed — revise if more rounds remain
            if round_num < self._max_rounds:
                logger.info(
                    "refinement_revising",
                    round=round_num,
                    suggestions=critique.suggestions,
                )
                feedback = self._format_feedback(critique)
                proposer_result = await self._proposer.analyze(
                    model_override=model_override,
                    critique_feedback=feedback,
                    **proposer_kwargs,
                )
                all_llm_calls.extend(proposer_result.llm_calls)
                proposal = proposer_result.output

        # Exhausted all rounds
        logger.warning("refinement_exhausted", max_rounds=self._max_rounds)
        return RefinementResult(
            proposal=proposal,
            critique=critique,
            rounds=self._max_rounds,
            exhausted=True,
            all_llm_calls=all_llm_calls,
        )

    @staticmethod
    def _format_feedback(critique: CritiqueResult) -> str:
        failed = [v for v in critique.verdicts if not v.passed]
        lines = [f"Previous proposal failed critique ({critique.summary}):"]
        for v in failed:
            lines.append(f"- [{v.dimension}] {v.reason}")
        if critique.suggestions:
            lines.append("\nSuggested improvements:")
            for s in critique.suggestions:
                lines.append(f"- {s}")
        return "\n".join(lines)
