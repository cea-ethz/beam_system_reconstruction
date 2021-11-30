import networkx as nx
import numpy as np
import open3d as o3d
import os
import progressbar
import shelve

from matplotlib import pyplot as plt
from tkinter import filedialog


import analysis_beams
import analysis_columns
import analysis_walls

import settings
import timer

import util_graph
import util_histogram
import util_cloud

# === TODO ===
# Refactor axs into global module
# Draw and export column images for debug / documentation

# === DEFINITIONS ===

DG = nx.DiGraph()

settings.write("do_dag_highlighting", False)

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

    settings.load_user_settings()

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
    timer.start("Read Cloud")
    pc_main = o3d.io.read_point_cloud(filepath)
    timer.end("Read Cloud")
    print(pc_main)
    #vis.add_geometry(cloud)

    # Calculate aabb for main cloud
    aabb_main = pc_main.get_axis_aligned_bounding_box()
    aabb_main.color = (1, 0, 0)
    if settings.read("visibility.world_aabb"):
        vis.add_geometry(aabb_main)

    # Add coordinate system to scene
    if settings.read("visibility.world_axis"):
        mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1000, origin=[0, 0, 0])
        vis.add_geometry(mesh_frame)

    # Setup histogram diagram
    px = 1 / plt.rcParams['figure.dpi']  # pixel in inches
    fig, axs = plt.subplots(2, 3, figsize=(1800*px, 1200*px))
    plt.tight_layout(h_pad=3.0)
    plt.subplots_adjust(left=0.05, bottom=0.10)

    plt.setp(axs[0, 0], ylabel="Normalized Point Density")
    plt.setp(axs[0, 0], xlabel='Z Axis Bins')
    plt.setp(axs[0, 1], xlabel='X Axis Bins')
    plt.setp(axs[0, 2], xlabel='Y Axis Bins')

    plt.setp(axs[1, 0], ylabel="Normalized Point Density")
    plt.setp(axs[1, 1], xlabel='X Axis Bins')
    plt.setp(axs[1, 2], xlabel='Y Axis Bins')

    # === Check for walls ===
    timer.start("Wall Analysis")
    pc_main = analysis_walls.analyze_walls(pc_main, aabb_main, axs, vis)
    timer.end("Wall Analysis")

    # === Perform main beam analysis ===
    timer.start("Beam Analysis")
    beam_layers, column_slice_positions, floor_levels = analysis_beams.detect_beams(pc_main, aabb_main, axs)

    # Finalize the histogram plots
    plt.savefig(dir_output + filename + "_plot.png")
    if settings.read("display.histogram"):
        timer.pause("Beam Analysis")
        plt.show()
        timer.unpause("Beam Analysis")
    else:
        plt.clf()

    # Split beams and add final forms to vis
    if len(beam_layers) != 2:
        print("Warning : {} beam layers, handling other than 2 beam layers not yet implemented".format(len(beam_layers)))
        beam_layers = beam_layers[-2:]
        print(len(beam_layers))

    primary_id = int(beam_layers[0].mean_spacing < beam_layers[1].mean_spacing)
    beam_layer_primary = beam_layers[primary_id]
    beam_layer_secondary = analysis_beams.perform_beam_splits(beam_layers[primary_id], beam_layers[int(not primary_id)], vis)

    if settings.read("visibility.beams_final"):
        for beam in beam_layer_primary.beams:
            vis.add_geometry(beam.cloud)
            vis.add_geometry(beam.aabb)
        for beam in beam_layer_secondary.beams:
            vis.add_geometry(beam.cloud)
            vis.add_geometry(beam.aabb)

    timer.end("Beam Analysis")

    # === Perform main column analysis ===
    timer.start("Column Analysis")
    for column_slice_position in column_slice_positions:
        pc_column = util_cloud.get_slice(pc_main, aabb_main, 2, column_slice_position, 1000, normalized=False)
        aabb_column = pc_column.get_axis_aligned_bounding_box()
        z_min = floor_levels[0] + 50 if len(floor_levels) else aabb_main.get_min_bound()[2]
        z_extents = (z_min, beam_layer_primary.average_z)
        columns = analysis_columns.analyze_columns(pc_column, aabb_column, pc_main, aabb_main, beam_layer_primary.beams, z_extents, vis)

        if settings.read("visibility.columns_final"):
            for column in columns:
                vis.add_geometry(column.pc)
                vis.add_geometry(column.aabb)
    timer.end("Column Analysis")

    # === Construct DAG Diagram ===
    timer.start("DAG Analysis")
    analysis_beams.analyze_beam_connections(beam_layer_primary, beam_layer_secondary, DG)

    for column in columns:
        #print("Beam id : {}".format(column.child_beams[0].id))
        DG.add_edges_from([(column.child_beams[0].id, column.id)])
        DG.nodes[column.id]['layer'] = 0
        DG.nodes[column.id]['source'] = 'column'

    # Create multipartite layout and reposition nodes for visibility
    pos = nx.multipartite_layout(DG, 'layer')

    column_ids = [column.id for column in columns if column.id in DG.nodes]
    primary_ids = [beam.id for beam in beam_layer_primary.beams if beam.id in DG.nodes]
    secondary_ids = [beam.id for beam in beam_layer_secondary.beams if beam.id in DG.nodes]

    pos = util_graph.normalize_position(DG, pos, column_ids)
    pos = util_graph.simplify_position(DG, pos, primary_ids)
    pos = util_graph.normalize_position(DG, pos, primary_ids)
    pos = util_graph.simplify_position(DG, pos, secondary_ids)
    pos = util_graph.normalize_position(DG, pos, secondary_ids)

    # Calculate downstream counts
    for column_id in column_ids:
        upstream, downstream = util_graph.get_stream_counts(DG, column_id)
        DG.nodes[column_id]['stream'] = upstream
    for primary_id in primary_ids:
        upstream, downstream = util_graph.get_stream_counts(DG, primary_id)
        DG.nodes[primary_id]['stream'] = upstream
    for secondary_id in secondary_ids:
        upstream, downstream = util_graph.get_stream_counts(DG, secondary_id)
        DG.nodes[secondary_id]['stream'] = upstream

    # Generate beam labels from downstream counts
    labels = nx.get_node_attributes(DG, 'stream')

    if settings.read("do_dag_highlighting"):
        # Highlight model elements for example
        beam_layer_secondary.beams[7].aabb.color = (1, 0, 0)
        beam_layer_primary.beams[1].aabb.color = (1, 0, 0)

        # Highlight example nodes and edges
        secondary_node_id = beam_layer_secondary.beams[7].id
        primary_node_id = beam_layer_primary.beams[1].id

        node_colors = ['blue'] * len(DG.nodes)
        node_colors[util_graph.get_node_id(DG, secondary_node_id)] = 'red'
        node_colors[util_graph.get_node_id(DG, primary_node_id)] = 'red'

        edge_colors = ['black'] * len(DG.edges)
        edge_colors[util_graph.get_edge_id(DG, secondary_node_id, primary_node_id)] = 'red'

        nx.draw(DG, pos, node_color=node_colors, edge_color=edge_colors, labels=labels, with_labels=True, node_size=450,font_color="white")
    else:
        nx.draw(DG, pos, labels=labels, with_labels=True, node_size=450, font_color="white")

    plt.savefig(dir_output + filename + "_graph.png")
    if settings.read("display.dag"):
        timer.pause("DAG Analysis")
        plt.show()
        timer.unpause("DAG Analysis")
    else:
        plt.clf()

    timer.end("DAG Analysis")

    vis.run()
    vis.destroy_window()

    timer.check_for_orphans()


# === Script entry ===
if __name__ == '__main__':
    main()
