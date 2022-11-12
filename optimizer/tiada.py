import math
from typing import List, Optional

import torch
from torch import Tensor
from torch.optim.optimizer import Optimizer


class TiAda(Optimizer):
    r"""Implements TiAda algorithm.
    Args:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float, optional): learning rate (default: 1e-2)
        lr_decay (float, optional): learning rate decay (default: 0)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
        eps (float, optional): term added to the denominator to improve
            numerical stability (default: 1e-10)
        foreach (bool, optional): whether foreach implementation of optimizer is used (default: None)
        maximize (bool, optional): maximize the params based on the objective, instead of
            minimizing (default: False)
        alpha (float): alpha parameter in TiAda
        opponent_optim (optional): If this optimizer is for x, provide the optimizer of y. If
            this optimizer is for y, set it to None.
    """

    def __init__(
        self,
        params,
        lr=1e-2,
        lr_decay=0,
        weight_decay=0,
        initial_accumulator_value=0,
        eps=1e-10,
        foreach: Optional[bool] = None,
        alpha=0.5,
        opponent_optim=None,
        compute_effective_stepsize=False,
        *,
        maximize: bool = False,
    ):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= lr_decay:
            raise ValueError("Invalid lr_decay value: {}".format(lr_decay))
        if not 0.0 <= weight_decay:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))
        if not 0.0 <= initial_accumulator_value:
            raise ValueError(
                "Invalid initial_accumulator_value value: {}".format(
                    initial_accumulator_value
                )
            )
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))

        defaults = dict(
            lr=lr,
            lr_decay=lr_decay,
            eps=eps,
            weight_decay=weight_decay,
            initial_accumulator_value=initial_accumulator_value,
            foreach=foreach,
            maximize=maximize,
        )

        self.alpha = alpha
        self.opponent_optim = opponent_optim
        # whether to compute effective_stepsize
        self.compute_effective_stepsize = compute_effective_stepsize 

        super(TiAda, self).__init__(params, defaults)

        # store the total_sum in the same device as the first parameter
        self.total_sum = self.param_groups[0]["params"][0].new_zeros(1)

        with torch.no_grad():
            for group in self.param_groups:
                for p in group["params"]:
                    state = self.state[p]
                    state["step"] = torch.tensor(0.0)
                    init_value = (
                        complex(initial_accumulator_value, initial_accumulator_value)
                        if torch.is_complex(p)
                        else initial_accumulator_value
                    )
                    state["sum"] = torch.full_like(
                        p, init_value, memory_format=torch.preserve_format
                    )

                    # Update total_sum
                    self.total_sum.add_(state["sum"].sum())

    def __setstate__(self, state):
        super().__setstate__(state)
        for group in self.param_groups:
            group.setdefault("foreach", None)
            group.setdefault("maximize", False)

        state_values = list(self.state.values())
        step_is_tensor = (len(state_values) != 0) and torch.is_tensor(
            state_values[0]["step"]
        )
        if not step_is_tensor:
            for s in state_values:
                s["step"] = torch.tensor(float(s["step"]))

    def share_memory(self):
        for group in self.param_groups:
            for p in group["params"]:
                state = self.state[p]
                state["sum"].share_memory_()

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step.
        Args:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None

        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        # Update sum of norms
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is not None:
                    if torch.is_complex(p) or p.grad.is_sparse:
                        raise NotImplementedError
                    grad = p.grad
                    state = self.state[p]
                    sq_grad = torch.mul(grad, grad)
                    state["sum"].add_(sq_grad)
                    self.total_sum.add_(sq_grad.sum())

        # calculate the ratio
        if self.opponent_optim is not None:
            ratio = self.total_sum.pow(self.alpha)
            ratio.div_(
                    torch.max(
                        ratio,
                        self.opponent_optim.total_sum.pow(self.alpha)
                        )
                    )
        else:
            ratio = 1

        for group in self.param_groups:
            lr = group["lr"]
            lr_decay=group["lr_decay"]
            weight_decay=group["weight_decay"]
            eps=group["eps"]
            maximize=group["maximize"]

            for p in group["params"]:
                if p.grad is not None:

                    state = self.state[p]
                    grad = p.grad
                    state_sum = state["sum"]

                    step_t = state["step"]
                    step_t += 1
                    step = step_t.item()
                    
                    grad = grad if not maximize else -grad

                    if weight_decay != 0:
                        if grad.is_sparse:
                            raise RuntimeError(
                                "weight_decay option is not compatible with sparse gradients"
                            )
                        grad = grad.add(p, alpha=weight_decay)

                    clr = lr / (1 + (step - 1) * lr_decay)
                    # already updated sum
                    ratio_p = state_sum.pow(self.alpha).add_(eps).div_(ratio)
                    p.addcdiv_(grad, ratio_p, value=-clr)

                    if self.compute_effective_stepsize:
                        self.effective_stepsize = (clr / ratio_p).item()

        return loss

class TiAda_wo_max(Optimizer):
    def __init__(
        self,
        params,
        lr=1e-2,
        lr_decay=0,
        weight_decay=0,
        initial_accumulator_value=0,
        eps=1e-10,
        foreach: Optional[bool] = None,
        alpha=0.5,
        *,
        maximize: bool = False,
    ):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= lr_decay:
            raise ValueError("Invalid lr_decay value: {}".format(lr_decay))
        if not 0.0 <= weight_decay:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))
        if not 0.0 <= initial_accumulator_value:
            raise ValueError(
                "Invalid initial_accumulator_value value: {}".format(
                    initial_accumulator_value
                )
            )
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))

        defaults = dict(
            lr=lr,
            lr_decay=lr_decay,
            eps=eps,
            weight_decay=weight_decay,
            initial_accumulator_value=initial_accumulator_value,
            foreach=foreach,
            maximize=maximize,
        )
        super(TiAda_wo_max, self).__init__(params, defaults)

        self.alpha = alpha

        with torch.no_grad():
            for group in self.param_groups:
                for p in group["params"]:
                    state = self.state[p]
                    state["step"] = torch.tensor(0.0)
                    init_value = (
                        complex(initial_accumulator_value, initial_accumulator_value)
                        if torch.is_complex(p)
                        else initial_accumulator_value
                    )
                    state["sum"] = torch.full_like(
                        p, init_value, memory_format=torch.preserve_format
                    )

    def __setstate__(self, state):
        super().__setstate__(state)
        for group in self.param_groups:
            group.setdefault("foreach", None)
            group.setdefault("maximize", False)

        state_values = list(self.state.values())
        step_is_tensor = (len(state_values) != 0) and torch.is_tensor(
            state_values[0]["step"]
        )
        if not step_is_tensor:
            for s in state_values:
                s["step"] = torch.tensor(float(s["step"]))

    def share_memory(self):
        for group in self.param_groups:
            for p in group["params"]:
                state = self.state[p]
                state["sum"].share_memory_()

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step.
        Args:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None

        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        # Update sum of norms
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is not None:
                    if torch.is_complex(p) or p.grad.is_sparse:
                        raise NotImplementedError
                    grad = p.grad
                    state = self.state[p]
                    sq_grad = torch.mul(grad, grad)
                    state["sum"].add_(sq_grad)

        for group in self.param_groups:
            lr = group["lr"]
            lr_decay=group["lr_decay"]
            weight_decay=group["weight_decay"]
            eps=group["eps"]
            maximize=group["maximize"]

            for p in group["params"]:
                if p.grad is not None:

                    state = self.state[p]
                    grad = p.grad
                    state_sum = state["sum"]

                    step_t = state["step"]
                    step_t += 1
                    step = step_t.item()
                    
                    grad = grad if not maximize else -grad

                    if weight_decay != 0:
                        if grad.is_sparse:
                            raise RuntimeError(
                                "weight_decay option is not compatible with sparse gradients"
                            )
                        grad = grad.add(p, alpha=weight_decay)

                    clr = lr / (1 + (step - 1) * lr_decay)
                    # already updated sum
                    ratio_p = state_sum.pow(self.alpha).add_(eps)
                    p.addcdiv_(grad, ratio_p, value=-clr)

        return loss



# Adam

class TiAda_Adam(Optimizer):
    r"""Implements TiAda-Adam algorithm.
    Args:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float, optional): learning rate (default: 1e-3)
        betas (Tuple[float, float], optional): coefficients used for computing
            running averages of gradient and its square (default: (0.9, 0.999))
        eps (float, optional): term added to the denominator to improve
            numerical stability (default: 1e-8)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
        amsgrad (boolean, optional): whether to use the AMSGrad variant of this
            algorithm from the paper `On the Convergence of Adam and Beyond`_
            (default: False)
        foreach (bool, optional): whether foreach implementation of optimizer
            is used (default: None)
        maximize (bool, optional): maximize the params based on the objective, instead of
            minimizing (default: False)
        capturable (bool, optional): whether this instance is safe to capture in a CUDA graph.
            Passing True can impair ungraphed performance, so if you don't intend to
            graph capture this instance, leave it False (default: False)
        alpha (float): alpha parameter in TiAda
        opponent_optim (optional): If this optimizer is for x, provide the optimizer of y. If
            this optimizer is for y, set it to None.
    """

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=0, amsgrad=False,
                 alpha=0.5,
                 opponent_optim=None,
                 *, foreach: Optional[bool] = None,
                 maximize: bool = False, capturable: bool = False):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        if not 0.0 <= weight_decay:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))
        defaults = dict(lr=lr, betas=betas, eps=eps,
                        weight_decay=weight_decay, amsgrad=amsgrad,
                        maximize=maximize, foreach=foreach, capturable=capturable)
        super(TiAda_Adam, self).__init__(params, defaults)

        self.alpha = alpha
        self.opponent_optim = opponent_optim

        # store the total_sum in the same device as the first parameter
        self.total_sum = self.param_groups[0]["params"][0].new_zeros(1)

        with torch.no_grad():
            for group in self.param_groups:
                for p in group["params"]:
                    state = self.state[p]
                    state['step'] = torch.zeros((1,), dtype=torch.float, device=p.device) \
                        if self.defaults['capturable'] else torch.tensor(0.)
                    # Exponential moving average of gradient values
                    state['exp_avg'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    # Exponential moving average of squared gradient values
                    state['exp_avg_sq'] = torch.zeros_like(p, memory_format=torch.preserve_format)

                    if group['amsgrad']:
                        # Maintains max of all exp. moving avg. of sq. grad. values
                        state['max_exp_avg_sq'] = torch.zeros_like(p, memory_format=torch.preserve_format)

                        # Update total
                        self.total_sum.add_(state["max_exp_avg_sq"].sum())
                    else:
                        self.total_sum.add_(state["exp_avg_sq"].sum())



    def __setstate__(self, state):
        super().__setstate__(state)
        for group in self.param_groups:
            group.setdefault('amsgrad', False)
            group.setdefault('maximize', False)
            group.setdefault('foreach', None)
            group.setdefault('capturable', False)
        state_values = list(self.state.values())
        step_is_tensor = (len(state_values) != 0) and torch.is_tensor(state_values[0]['step'])
        if not step_is_tensor:
            for s in state_values:
                s['step'] = torch.tensor(float(s['step']))

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step.

        Args:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        self._cuda_graph_capture_health_check()

        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        amsgrad = self.defaults['amsgrad']

        # set total to 0
        self.total_sum.zero_()

        # Update the states
        for group in self.param_groups:
            beta1, beta2 = group['betas']
            maximize = group['maximize']
            capturable = group['capturable']
            weight_decay = group['weight_decay']
            lr = group['lr']
            eps = group['eps']

            for p in group['params']:
                param = p
                if p.grad is not None:
                    if p.grad.is_sparse:
                        raise RuntimeError('Adam does not support sparse gradients, please consider SparseAdam instead')

                    state = self.state[p]

                    grad = param.grad if not maximize else -param.grad
                    exp_avg = state['exp_avg']
                    exp_avg_sq = state['exp_avg_sq']
                    step_t = state['step']

                    if capturable:
                        # assert param.is_cuda and step_t.is_cuda, "If capturable=True, params and state_steps must be CUDA tensors."
                        raise NotImplementedError

                    if weight_decay != 0:
                        grad = grad.add(param, alpha=weight_decay)

                    # Decay the first and second moment running average coefficient
                    exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                    exp_avg_sq.mul_(beta2).addcmul_(grad, grad.conj(), value=1 - beta2)

                    if amsgrad:
                        max_exp_avg_sq = state['max_exp_avg_sq']
                        # Maintains the maximum of all 2nd moment running avg. till now
                        torch.maximum(max_exp_avg_sq, exp_avg_sq, out=max_exp_avg_sq)

                        # Update total
                        self.total_sum.add_(max_exp_avg_sq.sum())
                    else:
                        self.total_sum.add_(exp_avg_sq.sum())

        # calculate the ratio
        if self.opponent_optim is not None:
            ratio = self.total_sum.pow(self.alpha)
            ratio.div_(
                    torch.max(
                        ratio,
                        self.opponent_optim.total_sum.pow(self.alpha)
                        )
                    )
        else:
            ratio = 1

        # Update parameters
        for group in self.param_groups:

            beta1, beta2 = group['betas']
            maximize = group['maximize']
            capturable = group['capturable']
            weight_decay = group['weight_decay']
            lr = group['lr']
            eps = group['eps']

            for p in group['params']:
                param = p
                if p.grad is not None:
                    if p.grad.is_sparse:
                        raise RuntimeError('Adam does not support sparse gradients, please consider SparseAdam instead')

                    state = self.state[p]

                    grad = param.grad if not maximize else -param.grad
                    exp_avg = state['exp_avg']
                    exp_avg_sq = state['exp_avg_sq']
                    step_t = state['step']

                    # update step
                    step_t += 1

                    if capturable:
                        raise NotImplementedError
                    else:
                        step = step_t.item()

                        bias_correction1 = 1 - beta1 ** step
                        bias_correction2 = 1 - beta2 ** step

                        step_size = lr / bias_correction1

                        bias_correction2_sqrt = math.sqrt(bias_correction2)

                        if amsgrad:
                            # Use the max. for normalizing running avg. of gradient
                            max_exp_avg_sq = state['max_exp_avg_sq']
                            denom = (max_exp_avg_sq.pow(self.alpha) / bias_correction2_sqrt).add_(eps)
                        else:
                            denom = (exp_avg_sq.pow(self.alpha) / bias_correction2_sqrt).add_(eps)

                        denom.div_(ratio)

                        param.addcdiv_(exp_avg, denom, value=-step_size)

        return loss


# Copy of AdaGrad, customized to compute the effective stepsize

class Adagrad(Optimizer):
    r"""Implements Adagrad algorithm.

    .. math::
       \begin{aligned}
            &\rule{110mm}{0.4pt}                                                                 \\
            &\textbf{input}      : \gamma \text{ (lr)}, \: \theta_0 \text{ (params)}, \: f(\theta)
                \text{ (objective)}, \: \lambda \text{ (weight decay)},                          \\
            &\hspace{12mm}    \tau \text{ (initial accumulator value)}, \: \eta\text{ (lr decay)}\\
            &\textbf{initialize} :  state\_sum_0 \leftarrow 0                             \\[-1.ex]
            &\rule{110mm}{0.4pt}                                                                 \\
            &\textbf{for} \: t=1 \: \textbf{to} \: \ldots \: \textbf{do}                         \\
            &\hspace{5mm}g_t           \leftarrow   \nabla_{\theta} f_t (\theta_{t-1})           \\
            &\hspace{5mm} \tilde{\gamma}    \leftarrow \gamma / (1 +(t-1) \eta)                  \\
            &\hspace{5mm} \textbf{if} \: \lambda \neq 0                                          \\
            &\hspace{10mm} g_t \leftarrow g_t + \lambda \theta_{t-1}                             \\
            &\hspace{5mm}state\_sum_t  \leftarrow  state\_sum_{t-1} + g^2_t                      \\
            &\hspace{5mm}\theta_t \leftarrow
                \theta_{t-1}- \tilde{\gamma} \frac{g_t}{\sqrt{state\_sum_t}+\epsilon}            \\
            &\rule{110mm}{0.4pt}                                                          \\[-1.ex]
            &\bf{return} \:  \theta_t                                                     \\[-1.ex]
            &\rule{110mm}{0.4pt}                                                          \\[-1.ex]
       \end{aligned}

    For further details regarding the algorithm we refer to `Adaptive Subgradient Methods for Online Learning
    and Stochastic Optimization`_.

    Args:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float, optional): learning rate (default: 1e-2)
        lr_decay (float, optional): learning rate decay (default: 0)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
        eps (float, optional): term added to the denominator to improve
            numerical stability (default: 1e-10)
        foreach (bool, optional): whether foreach implementation of optimizer is used (default: None)
        maximize (bool, optional): maximize the params based on the objective, instead of
            minimizing (default: False)

    .. _Adaptive Subgradient Methods for Online Learning and Stochastic
        Optimization: http://jmlr.org/papers/v12/duchi11a.html
    """

    def __init__(
        self,
        params,
        lr=1e-2,
        lr_decay=0,
        weight_decay=0,
        initial_accumulator_value=0,
        eps=1e-10,
        foreach: Optional[bool] = None,
        *,
        maximize: bool = False
    ):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= lr_decay:
            raise ValueError("Invalid lr_decay value: {}".format(lr_decay))
        if not 0.0 <= weight_decay:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))
        if not 0.0 <= initial_accumulator_value:
            raise ValueError(
                "Invalid initial_accumulator_value value: {}".format(
                    initial_accumulator_value
                )
            )
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))

        defaults = dict(
            lr=lr,
            lr_decay=lr_decay,
            eps=eps,
            weight_decay=weight_decay,
            initial_accumulator_value=initial_accumulator_value,
            foreach=foreach,
            maximize=maximize,
        )
        super(Adagrad, self).__init__(params, defaults)

        for group in self.param_groups:
            for p in group["params"]:
                state = self.state[p]
                state["step"] = torch.tensor(0.0)
                init_value = (
                    complex(initial_accumulator_value, initial_accumulator_value)
                    if torch.is_complex(p)
                    else initial_accumulator_value
                )
                state["sum"] = torch.full_like(
                    p, init_value, memory_format=torch.preserve_format
                )

    def __setstate__(self, state):
        super().__setstate__(state)
        for group in self.param_groups:
            group.setdefault("foreach", None)
            group.setdefault("maximize", False)

        state_values = list(self.state.values())
        step_is_tensor = (len(state_values) != 0) and torch.is_tensor(
            state_values[0]["step"]
        )
        if not step_is_tensor:
            for s in state_values:
                s["step"] = torch.tensor(float(s["step"]))

    def share_memory(self):
        for group in self.param_groups:
            for p in group["params"]:
                state = self.state[p]
                state["sum"].share_memory_()

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step.

        Args:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None

        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            params_with_grad = []
            grads = []
            state_sums = []
            state_steps = []

            has_sparse_grad = False
            for p in group["params"]:
                if p.grad is not None:
                    if p.grad.is_sparse:
                        has_sparse_grad = True
                    params_with_grad.append(p)
                    grads.append(p.grad)
                    state = self.state[p]
                    state_sums.append(state["sum"])
                    state_steps.append(state["step"])

            params = params_with_grad
            lr=group["lr"]
            weight_decay=group["weight_decay"]
            lr_decay=group["lr_decay"]
            eps=group["eps"]
            has_sparse_grad=has_sparse_grad
            maximize=group["maximize"]

            for (param, grad, state_sum, step_t) in zip(params, grads, state_sums, state_steps):
                # update step
                step_t += 1
                step = step_t.item()
                grad = grad if not maximize else -grad

                if weight_decay != 0:
                    if grad.is_sparse:
                        raise RuntimeError(
                            "weight_decay option is not compatible with sparse gradients"
                        )
                    grad = grad.add(param, alpha=weight_decay)

                clr = lr / (1 + (step - 1) * lr_decay)

                if grad.is_sparse:
                    grad = grad.coalesce()  # the update is non-linear so indices must be unique
                    grad_indices = grad._indices()
                    grad_values = grad._values()
                    size = grad.size()

                    state_sum.add_(_make_sparse(grad, grad_indices, grad_values.pow(2)))
                    std = state_sum.sparse_mask(grad)
                    std_values = std._values().sqrt_().add_(eps)
                    param.add_(
                        _make_sparse(grad, grad_indices, grad_values / std_values), alpha=-clr
                    )
                else:
                    is_complex = torch.is_complex(param)
                    if is_complex:
                        grad = torch.view_as_real(grad)
                        state_sum = torch.view_as_real(state_sum)
                        param = torch.view_as_real(param)
                    state_sum.addcmul_(grad, grad, value=1)
                    std = state_sum.sqrt().add_(eps)
                    param.addcdiv_(grad, std, value=-clr)
                    if is_complex:
                        param = torch.view_as_complex(param)
                        state_sum = torch.view_as_complex(state_sum)

                    self.effective_stepsize = (clr / std).item()
        return loss
