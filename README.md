  # CS787: Generative Artificial Intelligence  
  ## Image to Handwriting Generation with Transformers & Diffusion
    
    Indian Institute of Technology Kanpur  
    Course: CS787 – Generative Artificial Intelligence  
    Semester Project – Group E1  
    
    **Team**
    
    - Ashvin Patidar (220243)  
    - Hritvija Singh (220459)  
    - Khush Gupta (220526)  
    - Kshitij Bagga (220552)  
    - Shashwat Agarwal (221004)  
    
    **Instructors:** Prof. Arnab Bhattacharya, Prof. Subhajit Roy  
    **Date:** November 15, 2025  
    
    ---
    
  ## 1. Project Overview
    
    This project explores **offline handwriting generation** from typed text while preserving a writer’s **personal style** (slant, stroke flow, curvature, spacing, etc.).
    
    We implement and extend two complementary generative pipelines:
    
    1. **Enhanced Handwriting Transformer (HWT)**
       - Reproduction of the ICCV 2021 **Handwriting Transformer** baseline.
       - New **stroke-level curve tokenization** module that explicitly encodes the geometry of handwriting.
       - Fuses **CNN style features** with **curve-based style tokens** to improve stroke-level fidelity.
    
    2. **Style-Conditioned Diffusion Model (DDPM)**
       - A **UNet-based diffusion model** conditioned only on writer style (not text).
       - Focuses on generating realistic word-level handwriting images that match a writer’s style.
       - A separate **page renderer** lays out generated words to form full paragraphs.
    
    The code in this repository corresponds to the implementation described in  
    **`CS787_Project_Report.pdf`**.
    
    ---
    
  ## 2. Key Contributions (What this repo adds)
    
    From the course project perspective, the main contributions are:
    
    1. **Reproduction of Handwriting Transformer**
       - Faithful re-implementation of the original HWT architecture (style encoder, transformer, decoder).
       - Training under limited compute while achieving a reasonable approximation of the original results.
    
    2. **Stroke-Level Curve Tokenization**
       - Skeletonizes handwritten word images and fits **Bezier curves** to stroke contours.
       - Represents each stroke fragment as an 8D vector \[x₁, y₁, x₂, y₂, x₃, y₃, x₄, y₄\].
       - Encodes these curve vectors with a lightweight **Transformer** to obtain a compact **geometry-aware style embedding**.
    
    3. **Fusion of CNN + Curve Style Representations**
       - Combines:
         - Global visual style from a **ResNet-based CNN**.
         - Local stroke geometry from the **curve tokenizer**.
       - The fused embedding is used by the transformer-based generator, improving stroke continuity and character transitions.
    
    4. **Style-Conditioned Diffusion for Handwriting**
       - Implements a **DDPM** with:
         - Style encoder → 256D writer style vector.
         - UNet backbone with conditional injection of style into the mid-block.
         - **Classifier-free guidance** for controllable style strength.
       - Designed to generate writer-specific strokes with high visual fidelity.
    
    5. **Evaluation & Analysis**
       - Uses metrics such as **FID**, **Character Error Rate (CER)**, and **Word Error Rate (WER)** via an OCR model.
       - Compares:
         - Ground truth vs original HWT vs HWT + curve tokenizer.
       - Discusses training stability issues and limitations of the diffusion pipeline under course-level compute constraints.

---

## 3. Repository Structure

At the top level:
  
    CS787-main/
    ├── CS787_Project_Report.pdf        # Full project report (methodology, results, discussion)
    ├── Handwriting_Transformers/       # Enhanced HWT + curve tokenizer implementation
    └── DDPM/                           # Style-conditioned diffusion model & evaluation

3.1 Handwriting_Transformers/ (Enhanced HWT)
    Key components:
    
    text
    Copy code
    Handwriting_Transformers/
    ├── README.md               # Upstream HWT README (original project)
    ├── INSTALL.md              # Dataset & setup instructions for HWT
    ├── params.py               # Experiment & dataset configuration
    ├── train.py                # Training script for TRGAN/HWT
    ├── test.py                 # Inference / evaluation on validation data
    ├── demo.ipynb              # Demo on IAM/CVL dataset
    ├── demo_custom_handwriting.ipynb
    │                           # Demo on custom style & text
    ├── mytext.txt              # Input text for generation demos
    ├── output*.png             # Sample generated outputs
    ├── data/
    │   ├── create_data.py      # Dataset creation helpers
    │   ├── dataset.py          # Dataset & DataLoader definitions
    │   └── prepare_data.py     # Preprocessing & pickling
    └── models/
        ├── model.py            # TRGAN / Handwriting Transformer wrapper
        ├── transformer.py      # Transformer encoder/decoder
        ├── curve_tokenizer.py  # ★ Stroke-level curve tokenizer (new)
        ├── BigGAN_layers.py    
        ├── BigGAN_networks.py
        ├── OCR_network.py
        ├── networks.py
        ├── blocks.py
        └── inception.py
3.2 DDPM/ (Style-Conditioned Diffusion)
    Key components:
    
    text
    Copy code
    DDPM/
    ├── README.md               # Upstream/derived README
    ├── INSTALL.md              # Environment & dataset notes
    ├── requirements.txt        # Pinned Python dependencies
    ├── params.py               # General experiment settings
    ├── train.py                # (HWT-style training; not central here)
    ├── train_diffusion.py      # ★ Train style-conditioned DDPM
    ├── train_diffusion_finetune.py
    ├── sample_diffusion.py     # ★ Sample handwriting from trained DDPM
    ├── evaluate_diffusion.py   # ★ Compute FID, CER, WER on generated images
    ├── demo*.ipynb             # Diffusion demos
    ├── data/
    │   ├── dataset.py          # Dataset using IAM/CVL pickles
    │   ├── params1.py          # Dataset paths, NUM_WRITERS etc.
    │   └── image_utils1.py
    └── models/
        ├── model.py            # TRGAN (for compatibility)
        ├── diffusion.py        # Integration helpers
        ├── diffusion/
        │   ├── style_encoder.py    # ★ Style encoder → 256D vector
        │   ├── unet_small.py       # ★ UNet backbone for DDPM
        │   ├── simple_ddpm.py      # DDPM time-stepping logic
        │   ├── ema.py              # Exponential moving average of weights
        │   └── ddpm_adapter.py
        ├── OCR_network.py      # CRNN OCR for CER/WER
        ├── BigGAN_layers.py
        ├── BigGAN_networks.py
        └── transformer.py
## 4. Installation & Environment

    Note: This project assumes Python 3.x, PyTorch, and access to a GPU (recommended).
    Exact versions are indicated in DDPM/requirements.txt and the upstream HWT documentation.

4.1 Create and Activate Environment


    Example using conda:
    bash
    Copy code
    conda create -n cs787-handwriting python=3.10
    conda activate cs787-handwriting
    Install common dependencies (high-level):
    
    bash
    Copy code
    pip install torch torchvision torchaudio  # choose CUDA build as appropriate
    pip install numpy pandas scipy scikit-image scikit-learn tqdm pillow matplotlib
    pip install tensorboard torchmetrics lmdb einops regex python-dateutil pyyaml
    For more precise versions, refer to:
    
    DDPM/requirements.txt
    
    Upstream notes in Handwriting_Transformers/INSTALL.md

## 5. Datasets & Preprocessing


    Both pipelines assume access to offline handwriting datasets like IAM or CVL.

5.1 For Handwriting Transformer
    
    
    Follow instructions in:

    text
    Copy code
    Handwriting_Transformers/INSTALL.md
    In summary (upstream flow):
    
    Download dataset pickles (e.g., IAM-32.pickle, CVL-32.pickle) and english_words.txt.
    
    Place them under:
    
    text
    Copy code
    Handwriting_Transformers/files/
    Configure dataset in params.py:
    
    python
    Copy code
    DATASET = 'IAM'  # or 'CVL'
    
    if DATASET == 'IAM':
        DATASET_PATHS = 'files/IAM-32.pickle'
        NUM_WRITERS = 339
    elif DATASET == 'CVL':
        DATASET_PATHS = 'files/CVL-32.pickle'
        NUM_WRITERS = 283
    params.py also controls:
    
    EXP_NAME (experiment folder name)
    
    EPOCHS, batch_size
    
    Transformer depth, hidden dims, etc.

5.2 For Diffusion Model


    The diffusion dataset reuses the same pickled structure as the HWT:
    
    python
    Copy code
    {
      'train': { writer_id: [ {'img': PIL.Image, 'label': str}, ... ], ... },
      'test':  { writer_id: [ {'img': PIL.Image, 'label': str}, ... ], ... }
    }
    DDPM/data/params1.py points to the dataset:
    
    python
    Copy code
    DATASET = 'IAM'
    if DATASET == 'IAM':
        DATASET_PATHS = 'files/IAM-32.pickle'
        NUM_WRITERS = 339
    # ...
    Place the same files/*.pickle used by HWT into:
    
    text
    Copy code
    DDPM/files/
    (or adjust paths accordingly in params1.py and dataset.py).
  
  ## 6. Running the Enhanced Handwriting Transformer
  6.1 Training
  
  
    From the project root:
  
    bash
    Copy code
    cd Handwriting_Transformers
    python train.py
    What happens:
    
    params.py:init_project() creates:
    
    text
    Copy code
    saved_images/EXP_NAME/
        Real/
        Fake/
    The model (TRGAN with transformer + curve tokenizer) is trained on IAM/CVL.
    
    Generated samples (real/fake pairs) are periodically saved in saved_images/EXP_NAME.
    
    Key modules involved:
    
    models/curve_tokenizer.py:
    
    Skeletonizes style images.
    
    Extracts Bezier curve segments and encodes them to tokens.
    
    models/model.py:
    
    Wraps CNN style encoder, curve encoder, transformer, and FCN decoder.

6.2 Testing / Demo on Saved Models
    
    
    Once trained or using a pretrained checkpoint:
    
    Use test.py or the Jupyter notebooks:
    
    bash
    Copy code
    cd Handwriting_Transformers
    jupyter notebook
    # open demo.ipynb or demo_custom_handwriting.ipynb
    Typical workflow in the demos:
    
    Select a writer style (style images from the dataset or custom images).
    
    Edit mytext.txt with desired typed text.
    
    Run cells to generate the corresponding handwritten images in the chosen style.

## 7. Running the Style-Conditioned Diffusion Model
7.1 Training DDPM
        
        
        From the project root:
        
        bash
        Copy code
        cd DDPM
        
        python train_diffusion.py \
            --batch_size 8 \
            --epochs 10 \
            --timesteps 200 \
            --sample_steps 50 \
            --lr 2e-4 \
            --log_dir runs/ddpm \
            --ckpt_dir ckpt_ddpm \
            --sample_dir samples \
            --img_h 32 \
            --img_w 128
        What this script does:
        
        Loads TextDataset from DDPM/data/dataset.py (using IAM/CVL pickles).
        
        Passes style images through models.diffusion.StyleEncoder to get a 256D style vector.
        
        Runs a UNetSmall + SimpleDDPM diffusion model to denoise from Gaussian noise to handwriting images.
        
        Logs training loss and sample images to TensorBoard (--log_dir).
        
        Saves checkpoints in --ckpt_dir.
        
        Writes periodic sampled images into --sample_dir.
        
        Note: The report mentions that achieving perfectly stable training for the diffusion model under course compute limits was challenging; the DDPM results here are exploratory and not as stable as the transformer-based pipeline.

7.2 Sampling from a Trained DDPM
    
    
    Given a trained checkpoint:
    
    bash
    Copy code
    cd DDPM
    
    python sample_diffusion.py \
        --ckpt ckpt_ddpm/epoch_XX.pt \
        --style path/to/style_image.png \
        --out_dir samples_generated \
        --height 64 \
        --width 192
    This will:
    
    Load the saved DDPM model and style encoder.
    
    Encode the given style image.
    
    Generate a word-level handwriting image in that style.
    
    Save it as samples_generated/generated_sample.png.

7.3 Evaluation (FID, CER, WER)
    
    
    To evaluate generated images:
    
    Prepare directories:
    
    text
    Copy code
    real_folder/  # Ground truth images
    fake_folder/  # Generated images (HWT or DDPM)
    Run evaluation:
    
    bash
    Copy code
    cd DDPM
    
    python evaluate_diffusion.py \
        --real_dir path/to/real_folder \
        --fake_dir path/to/fake_folder \
        --ocr_ckpt path/to/ocr_checkpoint.pth \
        --out_dir eval_results
    This script:
    
    Computes FID between real & fake image sets.
    
    Uses a CRNN OCR model (models/OCR_network.py) to measure:
    
    Character Error Rate (CER)
    
    Word Error Rate (WER)
    
    Writes metrics and optionally example outputs to eval_results/.

## 8. Methodology Summary (Code ↔ Report Mapping)

Section 3.2 (Enhanced HWT) in the report


    ↔ Handwriting_Transformers/models/*, especially:
    
    curve_tokenizer.py
    
    transformer.py
    
    model.py

Section 3.3 (Style-Conditioned Diffusion) in the report


    ↔ DDPM/models/diffusion/*, train_diffusion.py, sample_diffusion.py

Page renderer logic (layouting lines and words)


    ↔ implemented around the sampling pipeline & post-processing; the report details the algorithm (left-to-right packing, line wrapping, spacing).

Results & analysis (Chapter 4)


    ↔ saved_images/, samples/, evaluate_diffusion.py, and generated plots/screenshots in the report.

## 9. Limitations & Future Work (as per project)


        Diffusion model training is sensitive and challenging with limited compute,
        making it hard to match the stability and quality of large-scale text-conditioned diffusion models.
        
        Curve tokenization currently relies on hand-designed skeletonization & Bezier fitting; more learned or differentiable approaches could be explored.
        
        Extending to paragraph-level content conditioning directly in the diffusion model is an open direction.
        
        Multi-script and low-resource handwriting (e.g., Indic scripts) are promising future applications.

## 10. Acknowledgements


        This project builds on and extends:
        
        Handwriting Transformers (ICCV 2021) – original codebase for transformer-based handwriting generation.
        
        Standard diffusion model components (DDPM, UNet, classifier-free guidance).
        
