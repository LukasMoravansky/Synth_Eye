# Synt.Eye: Empowering Vision-Based Industrial AI with Synthetic Data

<p align="center">
  <img src=https://github.com/rparak/Synth_Eye/blob/main/images/Logo_White.png width="800" height="400">
</p>

A modular platform designed to generate high-quality, photorealistic synthetic data that accurately replicates real-world environments, objects, and conditions, thereby enhancing the efficiency and performance of neural network training in the manufacturing sector.

The platform was developed as part of internal research activities at the Research and Innovation Center INTEMAC.

## TODO

- Improve the light configuration to better support rectangular objects.
- Add a standardized dataset template structure to the root of the repository.
- Remove the `time.sleep(10)` delay from `gen_synthetic_data.py`.
- Ensure proper handling of label `.txt` files:
  - If a label file already exists for a given index, delete or overwrite it instead of appending new bounding boxes to the existing file.

## Contributors

<table> <tr> <td align="center"> <a href="https://github.com/rparak"> <img src="https://avatars.githubusercontent.com/rparak" width="120px;" alt="Roman Parak"/><br /> <strong>Roman Parak</strong> </a><br /> </td> <td align="center"> <a href="https://github.com/LukasMoravansky"> <img src="https://avatars.githubusercontent.com/LukasMoravansky" width="120px;" alt="Lukas Moravansky"/><br /> <strong>Lukas Moravansky</strong> </a><br /> </td> </tr> </table>

## License
[MIT](https://choosealicense.com/licenses/mit/)
