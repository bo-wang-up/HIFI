import torch


class Attribute(object):
    def __init__(self, model, args):
        self.timesteps = args.Ns
        self.model = model
        self.optimizer = torch.optim.Adam(
            self.model.attributes,
            lr=args.outer_lr,
            betas=(0.5, 0.999),
            weight_decay=args.weight_decay
        )

        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, args.epochs, eta_min=0)

    def step(self, input_valid, target_valid):
        self.optimizer.zero_grad()
        outer_loss = self._backward_step(input_valid, target_valid)
        self.model.attributes_clamp()

        return outer_loss

    def _backward_step(self, input_valid, target_valid):
        output_valid = self.model(input_valid, timesteps=self.timesteps)
        loss = self.model.outer_loss(output_valid, target_valid)
        loss.backward()

        return loss
