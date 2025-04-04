# ------------------------------------------------------------------------------
# Created by Vasileios Tzouras, 2024
# ------------------------------------------------------------------------------

from torch.autograd import Function


class GradientReversalLayer(Function):
    """
    GradientReversalLayer function as described in the paper:
    "Unsupervised Domain Adaptation by Backpropagation" (Ganin & Lempitsky, 2015).

    This function is used for gradient reversal during domain adaptation.
    """

    @staticmethod
    def forward(ctx, x, alpha):
        """
        Forward pass of the GradientReversalLayer function.

        Args:
            ctx (torch.autograd.Function): Context.
            x (torch.Tensor): Input tensor.
            alpha (float): Scaling factor for gradient reversal.

        Returns:
            torch.Tensor: Input tensor unchanged.
        """
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        """
        Backward pass of the GradientReversalLayer function.

        Args:
            ctx (torch.autograd.Function): Context.
            grad_output (torch.Tensor): Gradient of the loss with respect to
                the output.

        Returns:
            torch.Tensor: Gradient of the loss with respect to the input.
            None: There is no gradient with respect to alpha.
        """
        output = grad_output.neg() * ctx.alpha
        return output, None
