import faiss
import torch
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
import datetime


def compute_features(
    geoloc_dataset,
    model,
    global_features_dim,
    num_workers=4,
    eval_batch_size=8,
    recall_values=[1, 5, 10, 100],
):

    test_dataloader = DataLoader(
        dataset=geoloc_dataset,
        num_workers=num_workers,
        batch_size=eval_batch_size,
        pin_memory=True,
    )
    model = model.eval()
    with torch.no_grad():
        gallery_features = np.empty(
            (len(geoloc_dataset), global_features_dim), dtype="float32"
        )
        local_features = []
        for inputs, indices in tqdm(
            test_dataloader, desc=f"Comp feats {geoloc_dataset}", ncols=120
        ):
            B, C, H, W = inputs.shape
            inputs = inputs.cuda()
            output = model("features_extractor", [inputs, "global"])
            local_features.append(output[1].cpu())
            output = output[0]
            output = output.reshape(B, global_features_dim)
            gallery_features[indices.detach().numpy(), :] = (
                output.detach().cpu().numpy()
            )
    query_features = gallery_features[geoloc_dataset.gallery_num :]
    gallery_features = gallery_features[: geoloc_dataset.gallery_num]
    local_features = torch.cat(local_features, dim=0)
    query_local = local_features[geoloc_dataset.gallery_num :]
    gallery_local = local_features[: geoloc_dataset.gallery_num]
    faiss_index = faiss.IndexFlatL2(global_features_dim)
    faiss_index.add(gallery_features)

    max_recall_value = max(recall_values)
    distances, predictions = faiss_index.search(query_features, max_recall_value)
    ground_truths = geoloc_dataset.get_positives()

    recalls, recalls_str = compute_recalls(
        predictions, ground_truths, geoloc_dataset, recall_values
    )

    correct_bool_mat = np.zeros(
        (geoloc_dataset.queries_num, max_recall_value), dtype=int
    )
    for query_index in range(geoloc_dataset.queries_num):
        positives = set(ground_truths[query_index].tolist())
        for pred_index in range(max_recall_value):
            pred = predictions[query_index, pred_index]
            if pred in positives:
                correct_bool_mat[query_index, pred_index] = 1
    return (
        recalls,
        recalls_str,
        predictions,
        correct_bool_mat,
        distances,
        query_local,
        gallery_local,
    )


def compute_recalls(
    predictions, ground_truths, test_dataset, recall_values=[1, 5, 10, 20]
):

    recalls = np.zeros(len(recall_values))
    for query_index, pred in enumerate(predictions):
        for i, n in enumerate(recall_values):
            if np.any(np.in1d(pred[:n], ground_truths[query_index])):
                recalls[i:] += 1
                break
    recalls = recalls / test_dataset.queries_num * 100
    recalls_pretty_str = ", ".join(
        [f"R@{val}: {rec:.1f}" for val, rec in zip(recall_values, recalls)]
    )
    return recalls, recalls_pretty_str
