import paddle
from collections import OrderedDict
from paddle.optimizer import Optimizer
from paddle.nn import Layer
from typing import Dict, List, Callable, Union

from few_shot.core import create_nshot_task_label


def replace_grad(parameter_gradients, parameter_name):
    def replace_grad_(Layer):
        return parameter_gradients[parameter_name]

    return replace_grad_


def meta_gradient_step(model: Layer,
                       optimiser: Optimizer,
                       loss_fn: Callable,
                       x: paddle.Tensor,
                       y: paddle.Tensor,
                       n_shot: int,
                       k_way: int,
                       q_queries: int,
                       order: int,
                       inner_train_steps: int,
                       inner_lr: float,
                       train: bool,
                       device: Union[str, paddle.device]):
    """
    Perform a gradient step on a meta-learner.

    # Arguments
        model: Base model of the meta-learner being trained
        optimiser: Optimiser to calculate gradient step from loss
        loss_fn: Loss function to calculate between predictions and outputs
        x: Input samples for all few shot tasks
        y: Input labels of all few shot tasks
        n_shot: Number of examples per class in the support set of each task
        k_way: Number of classes in the few shot classification task of each task
        q_queries: Number of examples per class in the query set of each task. The query set is used to calculate
            meta-gradients after applying the update to
        order: Whether to use 1st order MAML (update meta-learner weights with gradients of the updated weights on the
            query set) or 2nd order MAML (use 2nd order updates by differentiating through the gradients of the updated
            weights on the query with respect to the original weights).
        inner_train_steps: Number of gradient steps to fit the fast weights during each inner update
        inner_lr: Learning rate used to update the fast weights on the inner update
        train: Whether to update the meta-learner weights at the end of the episode.
        device: Device on which to run computation
    """
    data_shape = x.shape[2:]
    create_graph = (True if order == 2 else False) and train

    task_gradients = []
    task_losses = []
    task_predictions = []
    for meta_batch in x:
        # By construction x is a 5D tensor of shape: (meta_batch_size, n*k + q*k, channels, width, height)
        # Hence when we iterate over the first  dimension we are iterating through the meta batches
        x_task_train = meta_batch[:n_shot * k_way]
        x_task_val = meta_batch[n_shot * k_way:]

        # Create a fast model using the current meta model weights
        fast_weights = OrderedDict(model.named_parameters())

        # Train the model for `inner_train_steps` iterations
        for inner_batch in range(inner_train_steps):
            # Perform update of model weights
            y = create_nshot_task_label(k_way, n_shot).to(device)
            logits = model.functional_forward(x_task_train, fast_weights)
            loss = loss_fn(logits, y)
            # gradients = torch.autograd.grad(loss, fast_weights.values(), create_graph=create_graph)
            gradients = paddle.gradients(loss , fast_weights.values())

            # Update weights manually
            fast_weights = OrderedDict(
                (name, param - inner_lr * grad)
                for ((name, param), grad) in zip(fast_weights.cpu().numpy(), gradients)
            )

        # Do a pass of the model on the validation data from the current task
        y = create_nshot_task_label(k_way, q_queries).to(device)
        logits = model.functional_forward(x_task_val, fast_weights)
        loss = loss_fn(logits, y)
        loss.backward(retain_graph=True)

        # Get post-update accuracies
        y_pred = logits.softmax(axis=1)
        task_predictions.append(y_pred)

        # Accumulate losses and gradients
        task_losses.append(loss)
        # gradients = torch.autograd.grad(loss, fast_weights.values(), create_graph=create_graph)
        gradients = paddle.gradients(loss , fast_weights.values())
        named_grads = {name: g for ((name, _), g) in zip(fast_weights.cpu().numpy(), gradients)}
        task_gradients.append(named_grads)

    if order == 1:
        if train:
            sum_task_gradients = {k: paddle.stack([grad[k] for grad in task_gradients]).mean(axis=0)
                                  for k in task_gradients[0].keys()}
            hooks = []
            for name, param in model.named_parameters():
                hooks.append(
                    param.register_hook(replace_grad(sum_task_gradients, name))
                )

            model.train()
            optimiser.clear_grad()
            # Dummy pass in order to create `loss` variable
            # Replace dummy gradients with mean task gradients using hooks
            logits = model(paddle.cast(paddle.zeros((k_way, ) + data_shape) , 'float64'))
            loss = loss_fn(logits, create_nshot_task_label(k_way, 1))
            loss.backward()
            optimiser.step()

            for h in hooks:
                h.remove()

        return paddle.stack(task_losses).mean(), paddle.concat(task_predictions)

    elif order == 2:
        model.train()
        optimiser.clear_grad()
        meta_batch_loss = paddle.stack(task_losses).mean()

        if train:
            meta_batch_loss.backward()
            optimiser.step()

        return meta_batch_loss, paddle.concat(task_predictions)
    else:
        raise ValueError('Order must be either 1 or 2.')
