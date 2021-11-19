import math
import networkx as nx
import numpy as np
import open3d as o3d
import os
import progressbar
import scipy.signal as signal
import shelve
import time

from matplotlib import pyplot as plt
from tkinter import filedialog

import analysis_beams
import analysis_walls

import util_graph
import util_histogram
import util_cloud

from BIM_Geometry import Beam, BeamSystemLayer


# Compare to walls
# Make basic material judgement
# Detect columns


# === DEFINITIONS ===

aabb_main = None


DG = nx.DiGraph()

show_histogram = True
show_dag = False
do_highlighting = False
show_splits = True
show_short_beams = True

vis = None


def set_up_vector(vis):
    vis.get_view_control().set_up((0.001, 0.000, 0.9999))
    #vis.get_view_control().set_up((-1, 0.000, 0.0))


def setup_vis():
    vis = o3d.visualization.VisualizerWithKeyCallback()
    #vis.register_key_callback(83, save_view)
    vis.register_key_callback(69, set_up_vector)
    vis.create_window()
    vis.get_render_option().point_size = 1

    return vis


def main():
    global vis

    # Load last cloud location from settings if applicable
    initial_dirname = os.getcwd()
    with shelve.open("settings") as db:
        if 'initial_dirname' in db:
            initial_dirname = db['initial_dirname']

    filepath = filedialog.askopenfilename(initialdir=initial_dirname, title="Choose Input Cloud")
    dirname = os.path.dirname(filepath) + "/"
    basename = os.path.basename(filepath)
    filename = os.path.splitext(basename)[0]
    dir_output = dirname + filename + "/"

    with shelve.open("settings") as db:
        db['initial_dirname'] = dirname

    if not os.path.exists(dir_output):
        os.makedirs(dir_output)

    project_data = shelve.open(dir_output + filename)
    project_data["test"] = 2
    project_data.close()

    vis = setup_vis()

    # Load cloud from file
    cloud = o3d.io.read_point_cloud(filepath)
    print(cloud)
    #vis.add_geometry(cloud)

    # Calculate aabb for main cloud
    aabb_main = cloud.get_axis_aligned_bounding_box()
    aabb_main.color = (1, 0, 0)
    #vis.add_geometry(aabb_main)

    # Add coordinate system to scene
    mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1000, origin=[0, 0, 0])
    vis.add_geometry(mesh_frame)

    # Setup histogram diagram
    px = 1 / plt.rcParams['figure.dpi']  # pixel in inches
    fig, axs = plt.subplots(2, 3, figsize=(1800*px, 1200*px))
    plt.tight_layout(h_pad=3.0)
    plt.subplots_adjust(left=0.05,bottom=0.10)

    plt.setp(axs[0, 0], ylabel="Normalized Point Density")
    plt.setp(axs[0, 0], xlabel='Z Axis Bins')
    plt.setp(axs[0, 1], xlabel='X Axis Bins')
    plt.setp(axs[0, 2], xlabel='Y Axis Bins')

    plt.setp(axs[1, 0], ylabel="Normalized Point Density")
    plt.setp(axs[1, 1], xlabel='X Axis Bins')
    plt.setp(axs[1, 2], xlabel='Y Axis Bins')

    # Check for walls
    cloud = analysis_walls.analyze_walls(cloud, aabb_main, axs, vis)
    #vis.add_geometry(cloud)

    # Perform main beam analysis
    beam_layers, column_slice_positions = analysis_beams.detect_beams(cloud, aabb_main, axs)

    print(column_slice_positions)

    # Finalize the histogram plots
    plt.savefig(dir_output + filename + "_plot.png")
    if show_histogram:
        plt.show()
    else:
        plt.clf()

    # Split beams and add final forms to vis
    if len(beam_layers) > 2:
        print("Error : Handling more than 2 beam layers not yet implemented")
    primary_id = int(beam_layers[0].mean_spacing < beam_layers[1].mean_spacing)

    secondary = analysis_beams.perform_beam_splits(beam_layers[primary_id], beam_layers[int(not primary_id)], vis)

    for beam in beam_layers[primary_id].beams:
        vis.add_geometry(beam.cloud)
        vis.add_geometry(beam.aabb)
    for beam in secondary.beams:
        vis.add_geometry(beam.cloud)
        vis.add_geometry(beam.aabb)

    # Perform main column analysis

    for column_slice_position in column_slice_positions:
        pc_column = util_cloud.get_slice(cloud, aabb_main, 2, column_slice_position, 1000, normalized=False)
        aabb_column = pc_column.get_axis_aligned_bounding_box()
        vis.add_geometry(pc_column)
        vis.add_geometry(aabb_column)


    # === Construct DAG Diagram ===
    analysis_beams.analyze_beam_connections(beam_layers[primary_id], secondary, DG)

    # Rescale smaller layers for visibility
    pos = nx.multipartite_layout(DG, 'layer')
    for i, pb in enumerate(beam_layers[primary_id].beams):
        n = 1.0 * i / (len(beam_layers[primary_id].beams) - 1)
        pos[pb.id][1] = n * 2 - 1

    # Calculate downstream counts
    for beam in beam_layers[primary_id].beams:
        if beam.id not in DG.nodes:
            continue
        upstream, downstream = util_graph.get_stream_counts(DG, beam.id)
        DG.nodes[beam.id]['stream'] = upstream

    for beam in secondary.beams:
        if beam.id not in DG.nodes:
            continue
        upstream, downstream = util_graph.get_stream_counts(DG, beam.id)
        DG.nodes[beam.id]['stream'] = upstream

    # Generate beam labels from downstream counts
    labels = nx.get_node_attributes(DG, 'stream')

    if do_highlighting:
        # Highlight model elements for example
        secondary.beams[7].aabb.color = (1, 0, 0)
        beam_layers[primary_id].beams[1].aabb.color = (1, 0, 0)

        # Highlight example nodes and edges
        secondary_node_id = secondary.beams[7].id
        primary_node_id = beam_layers[primary_id].beams[1].id
        # column_node_id =

        node_colors = ['blue'] * len(DG.nodes)
        node_colors[util_graph.get_node_id(DG, secondary_node_id)] = 'red'
        node_colors[util_graph.get_node_id(DG, primary_node_id)] = 'red'

        edge_colors = ['black'] * len(DG.edges)
        edge_colors[util_graph.get_edge_id(DG, secondary_node_id, primary_node_id)] = 'red'

        nx.draw(DG, pos, node_color=node_colors, edge_color=edge_colors, labels=labels, with_labels=True, node_size=300)
    else:
        nx.draw(DG, pos, labels=labels, with_labels=True, node_size=300)

    plt.savefig(dir_output + filename + "_graph.png")
    if show_dag:
        plt.show()
    else:
        plt.clf()

    vis.run()
    vis.destroy_window()


# === Script entry ===
if __name__ == '__main__':
    main()
