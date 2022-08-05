# formation-maintance-ccta2022
 A simulation code for the paper entitled "Communication-Efficient Formation Maintenance for Multi-Robot System with a Safety Certificate" presented in The 2022 IEEE Conference on Control Technology and Applications (CCTA) 

Author: Anirudh Aynala, Made Widhi Surya Atman, and Azwirman Gusrialdi

## sim2D_FormationObstacle.py
The main script which simulates the scenario 1 in the paper. Four robots are maintaining square formation towards a goal position while avoiding 2 obstacles.

The experiment with Turtlebot Burger is executed with ROSTB_FormationObstacle.py. The result can be viewed below.

[![Four-robot Formation with Obstacle Avoidance](https://img.youtube.com/vi/Ke9bf71z-pQ/0.jpg)](https://www.youtube.com/watch?v=Ke9bf71z-pQ "Four-robot Formation with Obstacle Avoidance")



## sim2D_FormationAvoidance.py
The main script which simulates the scenario 2 in the paper. The 2 robot formations, with 2 robots each, are crossing path towards goals while avoiding collision with each other.

Similar with above, the experiment is executed with ROSTB_FormationAvoidance.py. The result can be viewed below. 

[![Two-robot Formations with Collision Avoidance](https://img.youtube.com/vi/HoJuWlrpWqA/0.jpg)](https://www.youtube.com/watch?v=HoJuWlrpWqA "Two-robot Formations with Collision Avoidance")
