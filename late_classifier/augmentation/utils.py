import pandas as pd


def augment_just_training_set_features(train_features, train_labels, aug_features, aug_labels):
    augmented_oids = aug_features.index.values
    aug_original_oids = [oid.split('_')[0] for oid in augmented_oids]
    aug_original_oids_set = set(aug_original_oids)
    train_oids_set = set(train_features.index.values)
    useful_original_oids = train_oids_set.intersection(aug_original_oids_set)
    selected_aug_oids = [
        oid for oid in augmented_oids if oid.split('_')[0] in useful_original_oids]

    new_features = pd.concat([train_features, aug_features.loc[selected_aug_oids]], axis=0)
    new_labels = pd.concat([train_labels, aug_labels.loc[selected_aug_oids]], axis=0)
    return new_features, new_labels
