import paddle


def categorical_accuracy(y, y_pred):
    """Calculates categorical accuracy.

    # Arguments:
        y_pred: Prediction probabilities or logits of shape [batch_size, num_categories]
        y: Ground truth categories. Must have shape [batch_size,]
    """
    # return paddle.equal(y_pred.argmax(axis=-1), y).sum().cpu().numpy()[0]/ y_pred.shape[0]
    # torch.eq(y_pred.argmax(dim=-1), y).sum().item() / y_pred.shape[0]
    # print(paddle.cast(paddle.equal(y_pred.argmax(axis=-1), y),'int32').sum().cpu().numpy())
    # print(paddle.equal(y_pred.argmax(axis=-1), y))
    return paddle.cast(paddle.equal(y_pred.argmax(axis=-1), y),'int32').sum().cpu().numpy() / y_pred.shape[0]

NAMED_METRICS = {
    'categorical_accuracy': categorical_accuracy
}
