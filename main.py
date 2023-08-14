from experiments import Experiment


def main() -> None:
    exp = Experiment(
        dataset="mock_dataset",
        timepoint=None,
        n_epochs=20,
        n_folds=5,
        validation_period=1,
        learning_rate=0.001,
        loss_weight=1.0,
        model_name="mock_model_experiment",
    )
    exp.train_model()
    exp.run_inference()
    exp.get_results_table()


if __name__ == "__main__":
    main()
