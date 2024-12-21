from pymongo import MongoClient


def rename_metric(experiment_name, old_metric_name, new_metric_name):
    uri = "mongodb://mongo_user:mongo_password@127.0.0.1:27017/?authSource=admin"
    try:
        with MongoClient(uri) as client:
            database = client["sacred"]
            runs = database["runs"]
            metrics = database["metrics"]
            experiments = runs.find({"experiment.name": experiment_name})

            for experiment in experiments:
                run_id = experiment.get("_id")
                query = {"run_id": run_id, "name": old_metric_name}
                name_update = {"$set": {"name": new_metric_name}}
                x = metrics.update_many(query, name_update)

    except Exception as e:
        raise Exception("Error updating data in Mongo: ", e)


def main():
    rename_metric("acc_no_play_vs_acc_all_play_sea_label_count",
                  "ensemble_none.avg_majority_test",
                  "acc_ensemble.none_play.avg_majority_accuracy")
    rename_metric("acc_no_play_vs_acc_all_play_sea_label_count",
                  "ensemble_all.avg_majority_test",
                  "acc_ensemble.all_play.avg_majority_accuracy")
    rename_metric("elo_no_play_vs_elo_all_play_sea_label_count",
                  "ensemble_none.average_majority_accuracy",
                  "elo_ensemble.none_play.avg_majority_accuracy")
    rename_metric("elo_no_play_vs_elo_all_play_sea_label_count",
                  "ensemble_none.average_best_rated_accuracy",
                  "elo_ensemble.none_play.avg_best_rated_accuracy")
    rename_metric("elo_no_play_vs_elo_all_play_sea_label_count",
                  "ensemble_all.average_majority_accuracy",
                  "elo_ensemble.all_play.avg_majority_accuracy")
    rename_metric("elo_no_play_vs_elo_all_play_sea_label_count",
                  "ensemble_all.average_best_rated_accuracy",
                  "elo_ensemble.all_play.avg_best_rated_accuracy")
    rename_metric("single_plus_validation_vs_accuracy_ensemble_sea_label_count",
                  "ensemble.avg_majority_test_accuracy",
                  "acc_ensemble.avg_majority_test_accuracy")
    rename_metric("single_plus_validation_vs_accuracy_ensemble_sea_label_count",
                  "ensemble.avg_majority_validation_accuracy",
                  "acc_ensemble.avg_majority_val_accuracy")
    rename_metric("single_vs_accuracy_ensemble_random_tree",
                  "ensemble.avg_majority_test_accuracy",
                  "acc_ensemble.avg_majority_test_accuracy")
    rename_metric("single_vs_accuracy_ensemble_random_tree",
                  "ensemble.avg_majority_validation_accuracy",
                  "acc_ensemble.avg_majority_val_accuracy")
    rename_metric("single_vs_accuracy_ensemble_random_tree_label_count",
                  "ensemble.avg_majority_test_accuracy",
                  "acc_ensemble.avg_majority_test_accuracy")
    rename_metric("single_vs_accuracy_ensemble_random_tree_label_count",
                  "ensemble.avg_majority_validation_accuracy",
                  "acc_ensemble.avg_majority_val_accuracy")
    rename_metric("single_vs_accuracy_ensemble_sea",
                  "ensemble.avg_majority_test_accuracy",
                  "acc_ensemble.avg_majority_test_accuracy")
    rename_metric("single_vs_accuracy_ensemble_sea",
                  "ensemble.avg_majority_validation_accuracy",
                  "acc_ensemble.avg_majority_val_accuracy")
    rename_metric("single_vs_accuracy_ensemble_sea_label_count",
                  "ensemble.avg_majority_test_accuracy",
                  "acc_ensemble.avg_majority_test_accuracy")
    rename_metric("single_vs_accuracy_ensemble_sea_label_count",
                  "ensemble.avg_majority_validation_accuracy",
                  "acc_ensemble.avg_majority_val_accuracy")
    rename_metric("single_vs_elo_ensemble_random_tree",
                  "ensemble.average_majority_accuracy",
                  "elo_ensemble.avg_majority_accuracy")
    rename_metric("single_vs_elo_ensemble_random_tree",
                  "ensemble.average_best_rated_accuracy",
                  "elo_ensemble.avg_best_rated_accuracy")
    rename_metric("single_vs_elo_ensemble_random_tree_label_count",
                  "ensemble.average_majority_accuracy",
                  "elo_ensemble.avg_majority_accuracy")
    rename_metric("single_vs_elo_ensemble_random_tree_label_count",
                  "ensemble.average_best_rated_accuracy",
                  "elo_ensemble.avg_best_rated_accuracy")
    rename_metric("single_vs_elo_ensemble_sea",
                  "ensemble.average_majority_accuracy",
                  "elo_ensemble.avg_majority_accuracy")
    rename_metric("single_vs_elo_ensemble_sea",
                  "ensemble.average_best_rated_accuracy",
                  "elo_ensemble.avg_best_rated_accuracy")
    rename_metric("single_vs_elo_ensemble_sea_label_count",
                  "ensemble.avg_majority_test_accuracy",
                  "elo_ensemble.avg_majority_accuracy")
    rename_metric("single_vs_elo_ensemble_sea_label_count",
                  "ensemble.avg_best_rated_test_accuracy",
                  "elo_ensemble.avg_best_rated_accuracy")
    rename_metric("single_vs_random_ensemble_sea_label_count",
                  "ensemble.average_majority_accuracy",
                  "ran_ensemble.avg_majority_accuracy")


if __name__ == '__main__':
    main()