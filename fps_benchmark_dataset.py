import torch
from torch.utils.data import DataLoader
from scene import Scene
from tqdm import tqdm
import matplotlib.pyplot as plt

from gaussian_renderer import render
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel, FlameGaussianModel


def render_set(dataset : ModelParams, name, iteration, views, gaussians, pipeline, background, n_iter, vis=False):
    print(f"\n==== {name} set ====")
    views_loader = DataLoader(views, batch_size=None, shuffle=False, num_workers=8)
    view = next(iter(views_loader))

    for i in range(3):
        print(f"\nRound {i+1}")
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in tqdm(range(n_iter)):
            if gaussians.binding != None:
                gaussians.select_mesh_by_timestep(view.timestep)
            rendering = render(view, gaussians, pipeline, background)["render"]
        end.record()
        torch.cuda.synchronize()
        elapsed_time = start.elapsed_time(end) / 1000
        print(f"Rendering {n_iter} images took {elapsed_time:.2f} s")
        print(f"FPS: {n_iter / elapsed_time:.2f}")

    if vis:
        print("\nVisualizing the rendering result")
        plt.imshow(rendering.permute(1, 2, 0).clip(0, 1).cpu().numpy())
        plt.show()
        
def render_sets(dataset : ModelParams, iteration : int, pipeline : PipelineParams, skip_train : bool, skip_val : bool, skip_test : bool, n_iter : int, vis=False):
    with torch.no_grad():
        if dataset.bind_to_mesh:
            gaussians = FlameGaussianModel(dataset.sh_degree)
        else:
            gaussians = GaussianModel(dataset.sh_degree)
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)

        bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        if not skip_train:
            render_set(dataset, "train", scene.loaded_iter, scene.getTrainCameras(), gaussians, pipeline, background, n_iter, vis)
        
        if not skip_val:
            render_set(dataset, "val", scene.loaded_iter, scene.getValCameras(), gaussians, pipeline, background, n_iter, vis)

        if not skip_test:
            render_set(dataset, "test", scene.loaded_iter, scene.getTestCameras(), gaussians, pipeline, background, n_iter, vis)

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_val", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--n_iter", default=500, type=int)
    parser.add_argument("--vis", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    args = get_combined_args(parser)
    print("Rendering " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    render_sets(model.extract(args), args.iteration, pipeline.extract(args), args.skip_train, args.skip_val, args.skip_test, args.n_iter, args.vis)