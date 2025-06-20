import warnings
warnings.filterwarnings("ignore")

import sys
import random

from open_clip import get_tokenizer, build_zero_shot_classifier, get_input_dtype, F17K_DISEASE_113_CLASSES, OPENAI_SKIN_TEMPLATES
from open_clip_train.precision import get_autocast

import numpy as np
import torch

try:
    import wandb
except ImportError:
    wandb = None

try:
    import torch.utils.tensorboard as tensorboard
except ImportError:
    tensorboard = None

try:
    import horovod.torch as hvd
except ImportError:
    hvd = None

from open_clip import create_model_and_transforms, get_tokenizer
from open_clip_train.distributed import init_distributed_device
from open_clip_train.params import parse_args


def random_seed(seed=42, rank=0):
    torch.manual_seed(seed + rank)
    np.random.seed(seed + rank)
    random.seed(seed + rank)

def predict_topk_diseases(image_path, model, classifier, tokenizer, classnames, transforms, device='cpu', args=None, k=10):
    """
    Predict top-k diseases for a single image
    
    Args:
        image_path: Path to the input image
        model: CLIP model instance
        classifier: Pre-built zero-shot classifier weights
        tokenizer: CLIP tokenizer instance
        classnames: List of disease class names
        transforms: Image preprocessing transforms
        device: Device to use for inference
        args: Arguments object containing precision settings (optional)
    
    Returns:
        List of tuples (disease_name, confidence_score) for top-k predictions
    """
    from PIL import Image
    import torch
    import torch.nn.functional as F
    
    # Set up autocast and input dtype if args provided
    autocast = get_autocast(args.precision, device_type=device.type if isinstance(device, torch.device) else device)
    input_dtype = get_input_dtype(args.precision)

    
    # Load and preprocess image
    image = Image.open(image_path)
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    # Apply transforms and add batch dimension
    image_tensor = transforms(image).unsqueeze(0).to(device=device, dtype=input_dtype)
    
    with torch.inference_mode():
        with autocast():
            # Get image features
            output = model(image=image_tensor)
            image_features = output['image_features'] if isinstance(output, dict) else output[0]
            
            # Compute logits
            logits = 100.0 * image_features @ classifier
            
            # Get probabilities
            probabilities = torch.softmax(logits, dim=1)
            
            # Get top-k predictions
            topk_probs, topk_indices = torch.topk(probabilities, k=k, dim=1)
    
    # Convert to CPU and extract values
    topk_probs = topk_probs.cpu().numpy()[0]
    topk_indices = topk_indices.cpu().numpy()[0]
    
    # Create results list
    results = []
    for i, (idx, prob) in enumerate(zip(topk_indices, topk_probs)):
        disease_name = classnames[idx]
        confidence = float(prob)
        results.append((disease_name, confidence))
    
    return results

def print_topk_predictions(image_path, model, classifier, tokenizer, classnames, transforms, device='cpu', args=None, k=10):
    """
    Print top-k disease predictions for a single image in a formatted way
    """
    predictions = predict_topk_diseases(image_path, model, classifier, tokenizer, classnames, transforms, device, args, k=k)
    
    print(f"\nTop-{k} Disease Predictions for: {image_path}")
    print("-" * 60)
    for i, (disease, confidence) in enumerate(predictions, 1):
        print(f"{i}. {disease:<40} {confidence:.4f} ({confidence*100:.2f}%)")
    print("-" * 60)
    
    return predictions


def main(args):
    args = parse_args(args)

    if torch.cuda.is_available():
        # This enables tf32 on Ampere GPUs which is only 8% slower than
        # float16 and almost as accurate as float32
        # This was a default in pytorch until 1.12
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False

    # fully initialize distributed device environment
    device = init_distributed_device(args)

    if isinstance(args.force_image_size, (tuple, list)) and len(args.force_image_size) == 1:
        # arg is nargs, single (square) image size list -> int
        args.force_image_size = args.force_image_size[0]

    random_seed(args.seed, 0)
    model_kwargs = {}
    model, preprocess_train, preprocess_val = create_model_and_transforms(
        args.model,
        args.pretrained,
        precision=args.precision,
        device=device,
        jit=args.torchscript,
        force_quick_gelu=args.force_quick_gelu,
        force_custom_text=args.force_custom_text,
        force_patch_dropout=args.force_patch_dropout,
        force_image_size=args.force_image_size,
        image_mean=args.image_mean,
        image_std=args.image_std,
        image_interpolation=args.image_interpolation,
        image_resize_mode=args.image_resize_mode,  # only effective for inference
        aug_cfg=args.aug_cfg,
        pretrained_image=args.pretrained_image,
        output_dict=True,
        cache_dir=args.cache_dir,
        **model_kwargs,
    )

    random_seed(args.seed, args.rank)
    args.save_logs = None
    args.wandb = None

    # initialize datasets
    tokenizer = get_tokenizer(args.model, cache_dir=args.cache_dir)

    # Pytorch 2.0 adds '_orig_mod.' prefix to keys of state_dict() of compiled models.
    # For compatibility, we save state_dict() of the original model, which shares the
    # weights without the prefix.
    original_model = model
    if args.torchcompile:

        if args.grad_checkpointing and args.distributed:
            # As of now (~PyTorch 2.4/2.5), compile + checkpointing but DDP optimizer must be disabled
            torch._dynamo.config.optimize_ddp = False

        model = torch.compile(original_model)

    classifier_f17k_113_disease = build_zero_shot_classifier(
        model,
        tokenizer=tokenizer,
        classnames=F17K_DISEASE_113_CLASSES,
        templates=OPENAI_SKIN_TEMPLATES,
        num_classes_per_batch=10,
        device=args.device,
        use_tqdm=False,
    )

    image_path = args.image_path

    # Print results nicely
    print_topk_predictions(image_path, model, classifier_f17k_113_disease, tokenizer, 
                          F17K_DISEASE_113_CLASSES, preprocess_val, args.device, args, k=args.k)

if __name__ == "__main__":
    main(sys.argv[1:])
