


params = {
    'text' : {
        'p_text' : {0.7: 'WORD', 0.25: 'LINE', 0.05: 'PARA'}, #{0.7: 'WORD', 0.25: 'LINE', 0.05: 'PARA'},
        'size' : [50, 10],
        'source' : 'newsgroup/newsgroup.txt',
    },
    'color' : {
        'source' : 'models/colors_new.cp',
        'merge_range' : (0.72, 0.88, 1.0),
        'color_dis' : 0 # 0
    },
    'depth' : {
        'range' : (0.1, 100) # (0.1,100)
    },
    'method' : {
        'version' : 'v4', # v2 | base | v3
        'region_reuse' : 3,
        'postprocess' : 'hw', # hw | None
        'shelter' : False,
        'overlap' : False, # no overlaping text instances
    },
    'generator' : {
        'save' : 'gen_data/joint_10f_909_large',
        'seed' : 18, # random seed
        'tasks' : None,#'gen_data/act_10f_813_base_1k/task.pkl', # 'data/models/tasks_act.pkl'
        'datasets' : ['data/backgrounds/activitynet.txt'], #'data/backgrounds/activitynet.txt', 'data/backgrounds/got10k.txt', 'data/backgrounds/ytvis.txt'],
        'num_workers' : 6,
        'mode' : 'random', # random | round
        'frame_itv' : 5
    }
}

