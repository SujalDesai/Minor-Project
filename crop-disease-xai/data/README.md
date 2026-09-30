# Data Directory Structure

- `raw/`: Contains original leaf image subdirectories downloaded from the PlantVillage dataset repository.
  - `Tomato_Healthy/`
  - `Tomato_Early_Blight/`
  - `Tomato_Late_Blight/`
  - `Potato_Early_Blight/`
- `processed/`: Contains preprocessed train/validation/test dataset splits if cached.

## Dataset Details
Source: PlantVillage Dataset (spMohanty/PlantVillage-Dataset)
Classes: 4 selected agricultural plant disease classes for initial milestone prototype.
Image resolution: 256x256 RGB images, resized to 224x224 during pipeline preprocessing.
