import os
import os

sql_dir_war = "/home/kh/hkh/data/weight/logs/rsl_rl/Bruce_flat"
dbtype_list = os.listdir(sql_dir_war)
dbtype_list.sort(key=lambda file: os.path.getctime(os.path.join(sql_dir_war, file)))
for dbtype in dbtype_list:
    if os.path.isfile(os.path.join(sql_dir_war,dbtype)):
        dbtype_list.remove(dbtype)
print(dbtype_list[-1])
path = os.path.join(sql_dir_war, dbtype_list[-1],"rewards")
print(path)
cmd =f"tensorboard --logdir={path} --samples_per_plugin=scalars=10000000000"
os.system(cmd)