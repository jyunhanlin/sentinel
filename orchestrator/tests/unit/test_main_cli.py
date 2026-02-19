from orchestrator.__main__ import parse_args


class TestParseArgs:
    def test_default_mode_is_run(self):
        args = parse_args([])
        assert args.command is None

    def test_eval_subcommand(self):
        args = parse_args(["eval"])
        assert args.command == "eval"

    def test_perf_subcommand(self):
        args = parse_args(["perf"])
        assert args.command == "perf"
