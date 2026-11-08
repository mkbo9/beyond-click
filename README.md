# AI Project Phase 1

## Task
We are building a binary classification model to predict whether a video's thumbnail belongs to the top 20% or bottom 20% of its channel's videos, measured by views relative to the channel's usual performance. In addition, we identify and describe recurring thumbnail styles and visual elements (e.g., faces, large text) that appear more often in top-performing videos.

## Model(s)
*   **Feature Extractor:** OpenCLIP ViT-B/32 (pretrained, frozen), producing a 512-dimensional embedding per thumbnail.
*   **Pattern Discovery:** PCA + HDBSCAN clustering of embeddings; clusters named by SmolVLM2-2.2B-Instruct; visual elements scored with CLIP zero-shot text prompts.
*   **Baseline:** Logistic Regression on element scores only, plus a shuffled-label control (expected AUC ≈ 0.5).
*   **Main Model:** Logistic Regression on CLIP embeddings, evaluated with leave-channels-out cross-validation (GroupKFold) using ROC AUC.

## Dataset(s)
*   **Custom Thumbnail Dataset:** About 1,000–1,500 thumbnails with view counts and upload dates from 10 channels in one niche (long-form videos aged 6–36 months).
*   **Source:** Collected via the YouTube Data API v3.
*   **Preprocessing:** Each video's views are divided by the median views of the channel's 20 nearest uploads; the top and bottom 20% per channel are labeled. Thumbnails are padded to square and resized to 224×224. Data is split by channel (train on 8, test on 2, rotated) so the model is always evaluated on unseen channels.

test [by M]
