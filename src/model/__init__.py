from logging import getLogger
from src.model.entry import model_selection

logger = getLogger()


def build_model(params):
    model = model_selection(params)
    logger.info(f"Number of parameters: {sum([p.numel() for p in model.parameters() if p.requires_grad])}")

    model.cuda()

    return model
