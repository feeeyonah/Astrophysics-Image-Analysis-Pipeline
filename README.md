# Astrophysics-Image-Analysis-Pipeline
Dark matter accounts for ~25% of the universe's energy density within the ΛCDM model, yet its particle nature remains unknown. This project searches for evidence of macro dark matter interactions in granite using machine learning.

Granite has an estimated cosmic exposure time of 100–500 million years, making it a natural large-scale detector. A macro particle passing through granite would ionize and melt surrounding material, leaving an elliptical "melt path" — a darker tunnel detectable on opposite faces of a slab with no known natural explanation.

Python (NumPy, Matplotlib) was used to simulate melt-path signatures across varied granite textures, generating a labeled synthetic dataset. A neural network was then trained to classify images as marked or unmarked based on features consistent with a macro interaction. Once sufficient accuracy is reached, the model will be applied to real granite images to search for genuine dark matter signatures.
