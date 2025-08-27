from .trainer import ModelTrainer



def run_experiment(model_fns, preprocessors, filepath, epochs, callbacks, batch_size, dropout_rate):
    experiments = []

    for data_variant, preprocessor in preprocessors.items():
        train_ds, val_ds = preprocessor.get_datasets(filepath)

        for model_name, model_fn in model_fns.items():
            print(f"Training model: {model_name} | data: {data_variant}")

            hyperparams = {"batch_size": batch_size}
            if model_name == "dropout":
                hyperparams["dropout_rate"] = dropout_rate

            trainer = ModelTrainer(
                model_fn=model_fn,
                train_ds=train_ds,
                val_ds=val_ds,
                model_name=model_name,
                data_variant=data_variant,
                epochs=epochs,
                hyperparameters=hyperparams,
                callbacks=callbacks
            )
            results = trainer.train()
            experiments.append(results)

    return experiments
