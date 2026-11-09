from pathlib import Path
from unittest.mock import MagicMock, patch

from agents.run_arl import run_arl


def test_run_arl_domain_query():
    # We mock out all the heavy external integrations
    with patch("agents.run_arl.build_vector_store"), \
         patch("agents.run_arl.LiteratureSearcher") as mock_searcher, \
         patch("agents.run_arl.LiteratureRetriever"), \
         patch("agents.run_arl.LiteratureSynthesizer") as mock_synthesizer, \
         patch("agents.run_arl.ExperimentDesigner"), \
         patch("agents.run_arl.ExperimentRunner"), \
         patch("agents.run_arl.MetricsCollector") as mock_collector, \
         patch("etft.run_logger.RunLogger"), \
         patch("feedback.rl_loop.feedback_router.FeedbackRouter") as mock_feedback_router, \
         patch("pathlib.Path.mkdir"), \
         patch("builtins.open"):

        # Set up a fake synthesis so the loop completes without error
        mock_synthesizer_instance = mock_synthesizer.return_value
        mock_brief = MagicMock()
        mock_brief.hypotheses = ["Test Hypothesis 1"]
        # Return a dictionary directly so json.dump works
        mock_brief.model_dump.return_value = {"dummy": "data"}
        mock_synthesizer_instance.synthesize.return_value = mock_brief

        # We need mock router to return a string
        mock_router_instance = mock_feedback_router.return_value
        mock_router_instance.build_rl_context_prefix.return_value = "fake prefix"

        mock_collector_instance = mock_collector.return_value
        mock_collector_instance.to_dict.return_value = {"success_rate": 1.0}

        mock_searcher_instance = mock_searcher.return_value
        mock_searcher_instance.search.return_value = []

        cfg = {"vector_store": {}, "agents": {}}
        output_dir = Path("/tmp/dummy_output")
        bottleneck = "depthwise_conv"

        # Test 1: domain="tinyml"
        run_arl(
            cfg=cfg,
            bottleneck=bottleneck,
            output_dir=output_dir,
            domain="tinyml",
        )

        expected_query = f"{bottleneck} ultra-low-power time series classification hardware-aware neural architecture search"
        mock_searcher_instance.search.assert_called_with(expected_query)

        # Test 2: no domain provided (fallback)
        run_arl(
            cfg=cfg,
            bottleneck=bottleneck,
            output_dir=output_dir,
        )

        expected_query_fallback = f"{bottleneck} deep learning improvement techniques"
        mock_searcher_instance.search.assert_called_with(expected_query_fallback)

        # Test 3: query_override takes precedence
        run_arl(
            cfg=cfg,
            bottleneck=bottleneck,
            output_dir=output_dir,
            domain="tinyml",
            query_override="custom override query"
        )
        mock_searcher_instance.search.assert_called_with("custom override query")
