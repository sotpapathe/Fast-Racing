roslaunch plan_manage ral-eval-occ-map.launch &
sleep 1
roslaunch traj_server traj_server.launch &
sleep 1
roslaunch Ctrl ctrl_md.launch &
sleep 1
roslaunch plan_manage visualize.launch
wait
