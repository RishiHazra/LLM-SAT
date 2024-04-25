import os
import matplotlib.pyplot as plt
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

  df = df.groupby(['n','alpha'])['is_sat'].sum().reset_index()

  df['is_sat'] = df['is_sat'] / df['is_sat'].max()

  fig, ax = plt.subplots()

  for key, grp in df.groupby(['n']):
      ax = grp.plot(ax=ax, kind='line', x='alpha', y='is_sat', label=key[0])

  plt.legend(loc='best')
  plot_path =os.path.join(dir, f"phase_transition{extension}")
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

  # compue the standard deviation
  df['std'] = df.groupby(['n', 'alpha'])['time'].transform('std')

  # take the average time
  df = df.groupby(['n','alpha']).agg({'time': 'mean', 'std': 'first'}).reset_index()

  fig, ax = plt.subplots()

  for key, grp in df.groupby(['n']):
      ax = grp.plot(ax=ax, kind='line', x='alpha', y='time', label=key)
      # plot also the std
      # ax.fill_between(grp['alpha'], grp['time'] - grp['std'], grp['time'] + grp['std'], alpha=0.2)

  # set y-axis to log scale
  ax.set_yscale('log')

  plt.legend(loc='best')
  plot_path =os.path.join(dir, f"times{extension}")
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

  # compue the standard deviation
  df['std'] = df.groupby(['alpha'])['time'].transform('std')

  # take the average time
  df = df.groupby(['alpha']).agg({'time': 'mean', 'std': 'first'}).reset_index()

  fig, ax = plt.subplots()

  # ax = df.plot(ax=ax, kind='line', x='alpha', y='time', label='time')
  ax = df.plot(ax=ax, kind='line', x='alpha', y='time', label='time')
  # plot also the std
  # ax.fill_between(df['alpha'], df['time'] - df['std'], df['time'] + df['std'], alpha=0.2)

  # set y-axis to log scale
  ax.set_yscale('log')
  # remove legend
  ax.get_legend().remove()
  plot_path =os.path.join(dir, f"times_alpha{extension}")
  plt.savefig(plot_path)
  # plt.show()

def sat_probs(dir):
  df_file = os.path.join(dir, 'sat_solved.csv')
  df = pd.read_csv(df_file)

  df['is_sat'] = df['is_sat'].astype(int)

  df = df.groupby(['alpha'])['is_sat'].sum().reset_index()

  df['is_sat'] = round(df['is_sat'] / df['is_sat'].max(), 2)

  # save the sat probabilities to a file
  df.to_csv(os.path.join(dir, 'sat_probs.csv'), index=False)
  
# Version plotting quantiles (problem is that the 3rd quantile can be smaller than the mean)
# def plot_time_alpha(dir, extension='.pdf', extra_dir=None):
#   df_file = os.path.join(dir, 'sat_solved.csv')
#   df = pd.read_csv(df_file)

#   # if extra_dir is not None merge the two dataframes
#   if extra_dir is not None:
#     df_extra_file = os.path.join(extra_dir, 'sat_solved.csv')
#     df_extra = pd.read_csv(df_extra_file)
#     df = pd.concat([df, df_extra], ignore_index=True)

#   df['time'] = df['time'].astype(float)

#   # compute the percentiles
#   df['lower'] = df.groupby(['alpha'])['time'].transform(lambda x: x.quantile(0.25))
#   df['upper'] = df.groupby(['alpha'])['time'].transform(lambda x: x.quantile(0.75))

#   # take the average time
#   df = df.groupby(['alpha']).agg({'time': 'mean', 'lower': 'first', 'upper': 'first'}).reset_index()

#   fig, ax = plt.subplots()

#   ax = df.plot(ax=ax, kind='line', x='alpha', y='time', label='time')
#   # plot also the percentiles
#   ax.fill_between(df['alpha'], df['lower'], df['upper'], alpha=0.2)

#   # set y-axis to log scale
#   ax.set_yscale('log')

#   # remove legend
#   ax.get_legend().remove()
#   plot_path =os.path.join(dir, f"times_alpha{extension}")
#   plt.savefig(plot_path)
#   # plt.show()

if __name__ == "__main__":
  import argparse
  parser = argparse.ArgumentParser()

  task_parsers = parser.add_subparsers(dest='task', help='Plotting procedures')
  phase_transition_parser = task_parsers.add_parser('phase_transition')
  time_parser = task_parsers.add_parser('time')
  table_parser = task_parsers.add_parser('table')
  
  phase_transition_parser.add_argument('sat_res_dir', type=str, help="Log directory")

  time_parser.add_argument('sat_res_dir', type=str, help="Log directory")

  table_parser.add_argument('sat_res_dir', type=str, help="Log directory")

  parser.add_argument('--extension', type=str, help="File extension", default='.pdf')
  parser.add_argument('--extra_dir', type=str, help="Extra log directory", default=None)

  args = parser.parse_args()

  if args.task == 'phase_transition':        
    plot_phase_transition(args.sat_res_dir, extension=args.extension, extra_dir=args.extra_dir)
  elif args.task == 'time':
    plot_time(args.sat_res_dir, extension=args.extension, extra_dir=args.extra_dir)
    plot_time_alpha(args.sat_res_dir, extension=args.extension, extra_dir=args.extra_dir)
  elif args.task == 'table':
    sat_probs(args.sat_res_dir)
  else:
    raise NotImplementedError("Unsupported plotting task")