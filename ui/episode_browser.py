import numpy as np
import pathlib
import sys
sys.path.append(str(pathlib.Path(__file__).parent.parent))  # noqa


from bokeh.plotting import figure, curdoc
from bokeh.models import ColumnDataSource, DataTable, TableColumn
from bokeh import layouts

from envs import MiniGrid

MAX_STEPS = 500
PARENT_DIR = './data/MiniGrid-MazeS7-v0'
# DATA_DIR = './data/minigrid_PlaygroundV0/20210203_1535'
DATA_DIR = list(reversed(sorted(pathlib.Path(PARENT_DIR).glob('*'))))[0]  # Last subdirectory
print(f'Browsing data from: {DATA_DIR}')

episode_data = {}


def to_rgba(img, alpha=255):
    rgba = np.zeros(img.shape[0:2], dtype=np.uint32)
    view = rgba.view(dtype=np.uint8).reshape(rgba.shape + (4,))
    view[:, :, 0:3] = np.flipud(img)
    view[:, :, 3] = np.flipud(alpha)
    return rgba


def load_episode(path):
    with pathlib.Path(path).open('rb') as f:
        fdata = np.load(f)
        for key in fdata:
            episode_data[key] = fdata[key]
        # print([(k, v.shape) for k, v in data.items()])


def load_files(input_dir):
    input_dir = pathlib.Path(input_dir)
    files = sorted(input_dir.glob('*'))
    return {
        'name': [f.name for f in files],
        'path': [str(f) for f in files]
    }


def load_steps(path):
    load_episode(path)
    data = {}
    data['reward'] = episode_data['reward']
    if len(episode_data['action'].shape) == 1:
        data['action'] = episode_data['action']  # Index
    else:
        data['action'] = episode_data['action'].argmax(axis=1)  # One-hot
    return data


def load_frame(ix):
    if 'map' in episode_data:
        return load_frame_minigrid(ix)

    img = episode_data['image'][ix]
    img = np.flip(img, 0)
    assert len(img.shape) == 3 and img.shape[-1] in [1, 3], f'Unsupported image shape {img.shape}'

    return {'image': [to_rgba(img)]}


def load_frame_minigrid(ix):
    map_ = episode_data['map_agent'][ix]
    # map_vis = np.ones_like(map_)
    map_vis = episode_data['map_vis'][ix]

    img = MiniGrid.render_map(map_)
    alpha = 255
    alpha = (np.clip(1.0 - (map_vis / MAX_STEPS), 0, 1) * 255).astype(np.uint8).T
    tile_size = img.shape[0] // alpha.shape[0]
    alpha = np.repeat(np.repeat(alpha, tile_size, axis=0), tile_size, axis=1)

    return {'image': [to_rgba(img, alpha)]}


def files_selected(src, ix):
    if not ix:
        return
    ix = ix[-1]
    path = src.data['path'][ix]
    steps_source.data = load_steps(path)


def steps_selected(src, ix):
    if not ix:
        return
    ix = ix[-1]
    frame_source.data = load_frame(ix)


files_source = ColumnDataSource(data={'name': [], 'path': []})
files_table = DataTable(
    source=files_source,
    columns=[TableColumn(field="name", title="name")],
    width=200,
    height=600,
    selectable=True
)
files_source.selected.on_change('indices', lambda attr, old, new: files_selected(files_source, new))  # pylint: disable=no-member
files_source.data = load_files(DATA_DIR)


steps_source = ColumnDataSource(data={'action': [], 'reward': []})
steps_table = DataTable(
    source=steps_source,
    columns=[TableColumn(field=key, title=key) for key in steps_source.data],
    width=200,
    height=600,
    selectable=True
)
steps_source.selected.on_change('indices', lambda attr, old, new: steps_selected(steps_source, new))  # pylint: disable=no-member


frame_source = ColumnDataSource(data={'image': []})
frame_figure = figure(
    plot_width=600,
    plot_height=600,
    x_range=[0, 19],
    y_range=[0, 19]
)
fig = frame_figure
fig.image_rgba(image='image', x=0, y=0, dw=19, dh=19, source=frame_source)


curdoc().add_root(
    layouts.row(
        files_table,
        steps_table,
        frame_figure
    ))
