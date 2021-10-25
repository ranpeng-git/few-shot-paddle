import paddle


def categorical_accuracy(y, y_pred):
    """Calculates categorical accuracy.

    # Arguments:
        y_pred: Prediction probabilities or logits of shape [batch_size, num_categories]
        y: Ground truth categories. Must have shape [batch_size,]
    """
    return paddle.equal(y_pred.argmax(axis=-1), y).sum().cpu().numpy()[0]/ y_pred.shape[0]


NAMED_METRICS = {
    'categorical_accuracy': categorical_accuracy
}
