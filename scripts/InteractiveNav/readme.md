## 
scripts/InteractiveNav/read_scene_room_properties.py

```
conda activate mlspaces

python scripts/InteractiveNav/read_scene_room_properties.py \
  --scene_dataset procthor-10k \
  --data_split train \
  --house_ind 0 \
  --variant base \
  --room 2

python scripts/InteractiveNav/read_scene_room_properties.py --house_ind 1

python scripts/InteractiveNav/read_scene_room_properties.py --room 2 --background_mode bounds

```


