import os
import numpy as np
from scipy.interpolate import griddata
import matplotlib.pyplot as plt
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import pandas as pd


def plot_phase_transition(dir, extension='.pdf', extra_dir=None):
    df_file = os.path.join(dir, 'sat_solved.csv')
    df = pd.read_csv(df_file)

    # if extra_dir is not None merge the two dataframes
    if extra_dir is not None:
        df_extra_file = os.path.join(extra_dir, 'sat_solved.csv')
        df_extra = pd.read_csv(df_extra_file)
        df = pd.concat([df, df_extra], ignore_index=True)

    df['is_sat'] = df['is_sat'].astype(int)

    df = df.groupby(['n', 'alpha'])['is_sat'].sum().reset_index()

    df['is_sat'] = df['is_sat'] / df['is_sat'].max()

    fig, ax = plt.subplots()

    for key, grp in df.groupby(['n']):
        ax = grp.plot(ax=ax, kind='line', x='alpha', y='is_sat', label=key[0])

    plt.legend(loc='best')
    plot_path = os.path.join(dir, f"phase_transition{extension}")
    plt.savefig(plot_path)
    # plt.show()


# define a function that plots the time taken to solve a problem
def plot_time(dir, extension='.pdf', extra_dir=None):
    df_file = os.path.join(dir, 'sat_solved.csv')
    df = pd.read_csv(df_file)

    # if extra_dir is not None merge the two dataframes
    if extra_dir is not None:
        df_extra_file = os.path.join(extra_dir, 'sat_solved.csv')
        df_extra = pd.read_csv(df_extra_file)
        df = pd.concat([df, df_extra], ignore_index=True)

    df['time'] = df['time'].astype(float)

    # take the average time
    df = df.groupby(['n', 'alpha'])['time'].mean().reset_index()

    # compue the standard deviation
    df['std'] = df.groupby(['n'])['time'].transform('std')

    fig, ax = plt.subplots(figsize=(10, 8))

    for key, grp in df.groupby(['n']):
        ax = grp.plot(ax=ax, kind='line', x='alpha', y='time', label=key[0])
        # plot also the std
        ax.fill_between(grp['alpha'], grp['time'] - grp['std'], grp['time'] + grp['std'], alpha=0.2)

    plt.legend(loc='best')
    plot_path = os.path.join(dir, f"times{extension}")
    plt.savefig(plot_path)
    # plt.show()


# define a function that plots the time taken to solve a problem for a given alpha without distinguishing n
def plot_time_alpha(dir, extension='.pdf', extra_dir=None):
    df_file = os.path.join(dir, 'sat_solved.csv')
    df = pd.read_csv(df_file)

    # if extra_dir is not None merge the two dataframes
    if extra_dir is not None:
        df_extra_file = os.path.join(extra_dir, 'sat_solved.csv')
        df_extra = pd.read_csv(df_extra_file)
        df = pd.concat([df, df_extra], ignore_index=True)

    df['time'] = df['time'].astype(float)

    # take the average time
    df1 = df.groupby('alpha')['time'].mean().reset_index()

    # compute the standard deviation
    df1['std'] = df.groupby('alpha')['time'].std().tolist()
    print(df1['std'])
    fig, ax = plt.subplots(figsize=(10, 8))

    ax = df1.plot(ax=ax, kind='line', x='alpha', y='time', label='time')
    # plot also the std
    ax.fill_between(df1['alpha'],
                    df1['time'] - df1['std'],
                    df1['time'] + df1['std'],
                    alpha=0.2)

    # remove legend
    ax.get_legend().remove()
    plt.xlabel('alpha', fontsize=16)
    plt.ylabel('time', fontsize=16)
    plt.xticks(fontsize=13)
    plt.yticks(fontsize=13)
    plt.tight_layout()
    # plt.title('Time taken by Solver', fontsize=18)
    plot_path = os.path.join(dir, f"times_alpha{extension}")
    plt.savefig(plot_path)
    # plt.show()


def plot_times_3d(base_dir):
    df_file = os.path.join(base_dir, 'sat_solved.csv')
    time_3d_df = pd.read_csv(df_file)
    time_3d_df = time_3d_df[time_3d_df['alpha'] <= 11]

    time_3d_df['is_sat'] = time_3d_df['is_sat'].astype(int)
    time_3d_df = time_3d_df.groupby(['n', 'alpha'])['is_sat'].sum().reset_index()
    time_3d_df['is_sat'] = time_3d_df['is_sat'] / time_3d_df['is_sat'].max()

    # fig, ax = plt.subplots()

    # for key, grp in time_3d_df.groupby(['n']):
    #     ax = grp.plot(ax=ax, kind='line', x='alpha', y='is_sat', label=key[0])

    # time_3d_df = df.groupby(['alpha', 'num_variables'])['correct'].mean().reset_index(name='accuracy')
    azimuth = 95  # The angle to rotate around the z-axis
    elevation = 20
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    # time_3d_df['rolling_avg_accuracy'] = time_3d_df['accuracy'].rolling(window=3).mean().fillna(method='bfill')
    # We need to create a regular grid where each model_count and num_vars are represented
    # Let's create an interpolation grid for the model_count and num_vars values
    alpha_i = np.linspace(time_3d_df['alpha'].min(),
                          time_3d_df['alpha'].max(),
                          len(time_3d_df['alpha'].unique()))
    num_vars_i = np.linspace(time_3d_df['n'].min(),
                             time_3d_df['n'].max(),
                             len(time_3d_df['n'].unique()))
    alpha_ii, num_vars_ii = np.meshgrid(alpha_i, num_vars_i)

    # Interpolating; this will fill in the gaps in the rolling_avg_accuracy on the new grid
    time_i = griddata((time_3d_df['alpha'],
                           time_3d_df['n']),
                          time_3d_df['is_sat'],
                          (alpha_ii, num_vars_ii), method='cubic')

    surf = ax.plot_surface(alpha_ii, num_vars_ii, time_i, cmap='viridis', edgecolor='none')
    ax.view_init(elev=elevation, azim=azimuth)

    ax.zaxis.set_tick_params(length=0)

    # Increase the space between the z-axis title and the ticks
    ax.zaxis.labelpad = 30
    # Remove the black tick lines for all axes
    ax.xaxis.line.set_lw(0.)
    ax.yaxis.line.set_lw(0.)
    ax.zaxis.line.set_lw(0.)

    # Move the ticks to the left of the z-axis
    ax.zaxis.set_tick_params(pad=15)

    # # Setting the new limits
    ax.set_xlim(alpha_ii.max(), alpha_ii.min())
    ax.set_ylim(time_3d_df['n'].max(), time_3d_df['n'].min())
    ax.set_zlim(0, time_i.max())

    # Customize the z axis.
    # ax.set_zlim(0, 1.0)
    ax.zaxis.set_major_locator(LinearLocator(10))
    ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

    # Customizing the axes tick labels
    ax.tick_params(axis='both', which='major', labelsize=15)

    # Add a color bar which maps values to colors.
    # fig.colorbar(surf, shrink=0.5, aspect=5)

    # plt.grid(True)
    ax.set_ylabel('# variables', fontsize=15, labelpad=20)
    ax.set_xlabel('alpha', fontsize=15, labelpad=20)
    ax.set_zlabel('SAT probability', fontsize=15, labelpad=30)
    # Rotate the z-axis label
    ax.zaxis.set_rotate_label(False)  # This disables automatic rotation
    ax.zaxis.label.set_rotation(90)
    plt.tight_layout()
    # plt.show()
    plt.savefig(f'solver_phase_transitions.png')


if __name__ == "__main__":
    plot_times_3d('~/PycharmProjects/LLM-SAT')
    # import argparse
    #
    # parser = argparse.ArgumentParser()
    #
    # task_parsers = parser.add_subparsers(dest='task', help='Plotting procedures')
    # phase_transition_parser = task_parsers.add_parser('phase_transition')
    # time_parser = task_parsers.add_parser('time')
    #
    # phase_transition_parser.add_argument('sat_res_dir', type=str, help="Log directory")
    #
    # time_parser.add_argument('sat_res_dir', type=str, help="Log directory")
    #
    # parser.add_argument('--extension', type=str, help="File extension", default='.png')
    # parser.add_argument('--extra_dir', type=str, help="Extra log directory", default=None)
    #
    # args = parser.parse_args()
    #
    # if args.task == 'phase_transition':
    #     plot_phase_transition(args.sat_res_dir, extension=args.extension, extra_dir=args.extra_dir)
    # elif args.task == 'time':
    #     plot_time(args.sat_res_dir, extension=args.extension, extra_dir=args.extra_dir)
    #     plot_time_alpha(args.sat_res_dir, extension=args.extension, extra_dir=args.extra_dir)
    # else:
    #     raise NotImplementedError("Unsupported plotting task")
