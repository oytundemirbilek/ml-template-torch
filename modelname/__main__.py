"""Entrypoint for the CLI."""

from __future__ import annotations

from argparse import ArgumentParser, Namespace

from modelname import __version__
from modelname.experiments import Experiment


def parse_args() -> Namespace:
    """Parse command line arguments and return as dictionary."""
    parser = ArgumentParser(
        prog="modelname",
        description="Predict a target brain graph using a source brain graph.",
    )
    parser.add_argument(
        "-v", "--version", action="version", version=f"%(prog)s {__version__}"
    )
    main_args = parser.add_argument_group("main options")
    infer_args = parser.add_argument_group("test options")
    data_args = parser.add_argument_group("dataset options")
    eval_args = parser.add_argument_group("evaluation options")
    main_args.add_argument(
        "-m",
        "--model-name",
        type=str,
        default="default_model_name",
        help="model name to be loaded and used for testing/inference.",
    )
    main_args.add_argument(
        "-de",
        "--device",
        type=str,
        default=None,
        help="The device that model will use.",
        choices=["cpu", "cuda"],
    )
    main_args.add_argument(
        "-t",
        "--train",
        action="store_true",
        help="run the training loop on the target dataset.",
    )
    main_args.add_argument(
        "-i",
        "--infer",
        action="store_true",
        help="run inference loop on the target dataset for either testing or inference purposes.",
    )

    infer_args.add_argument(
        "-l",
        "--load-model",
        action="store_true",
        help="whether to load from the pretrained model files, in --infer mode.",
    )
    infer_args.add_argument(
        "-is",
        "--save-each",
        action="store_true",
        help="whether to store every predicted graph for each subject, in --infer mode.",
    )
    infer_args.add_argument(
        "-f",
        "--fold-id",
        type=int,
        default=None,
        help="from which dataset fold the model is trained on, in --infer mode.",
    )
    infer_args.add_argument(
        "-r",
        "--get-table",
        action="store_true",
        help="whether to store results in a table at the end, in --infer mode.",
    )

    data_args.add_argument(
        "-d",
        "--dataset",
        type=str,
        default="mock_dataset",
        help="dataset name which the operations will be based on.",
        choices=["mock_dataset1", "mock_dataset2"],
    )

    eval_args.add_argument(
        "-em",
        "--eval-metric",
        type=str,
        default="mse",
        help="which evaluation metric should be used when plotting or statistical testing.",
        choices=["mse", "l1"],
    )

    return parser.parse_args()


def main() -> None:
    """Run main function from CLI."""
    args = parse_args()

    exp = Experiment(
        dataset=args.dataset,
        timepoint=None,
        n_epochs=20,
        n_folds=5,
        validation_period=1,
        learning_rate=0.001,
        loss_weight=1.0,
        model_name=args.model_name,
        device=args.device,
    )
    if args.train:
        exp.train_model()
    if args.infer:
        exp.run_inference(
            mode="test",
            load=args.load_model,
            save_predictions=args.save_each,
            fold=args.fold_id,
        )
    if args.get_table:
        exp.get_results_table()


if __name__ == "__main__":
    main()
