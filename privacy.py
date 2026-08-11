import torch


def add_differential_privacy(
    model,
    clip_value=1.0,
    noise_scale=0.01
):
    """
    Applies simple differential privacy to model parameters.

    1. Clips each parameter tensor.
    2. Adds Gaussian noise.
    """

    with torch.no_grad():

        for parameter in model.parameters():

            # Clip parameter values
            parameter.clamp_(
                -clip_value,
                clip_value
            )

            # Add Gaussian noise
            noise = torch.normal(
                mean=0.0,
                std=noise_scale,
                size=parameter.shape
            )

            parameter.add_(noise)

    return model