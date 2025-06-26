clear; clc;

syms t1 t2 t3 t4 t5 t6 real
syms theta alpha a d real

% DH table:  [θ   d         a        α]
DH = [ t1+pi     0.1273     0        pi/2;
       t2        0          -0.612   0;
       t3        0          -0.5723  0 ;
       t4        0.163941   0        pi/2;
       t5        0.1157     0        -pi/2;
       t6        0.0922     0        0];

% Symbolic elementary transform
Telem = [ cos(theta)  -sin(theta)*cos(alpha)   sin(theta)*sin(alpha)   a*cos(theta);
          sin(theta)   cos(theta)*cos(alpha)  -cos(theta)*sin(alpha)   a*sin(theta);
          0                sin(alpha)              cos(alpha)              d;
          0                    0                       0                   1 ];

row = 1; 
params = DH(row, :);
T1 = vpa( subs(Telem, [theta d a alpha], params) );

row = 2; 
params = DH(row, :);
T2 = vpa( subs(Telem, [theta d a alpha], params) );

row = 3; 
params = DH(row, :);
T3 = vpa( subs(Telem, [theta d a alpha], params) );

row = 4; 
params = DH(row, :);
T4 = vpa( subs(Telem, [theta d a alpha], params) );

row = 5; 
params = DH(row, :);
T5 = vpa( subs(Telem, [theta d a alpha], params) );

row = 6; 
params = DH(row, :);
T6 = vpa( subs(Telem, [theta d a alpha], params) );

T01 = T1;
T02 = T1 * T2;
T03 = T02 * T3;
T04 = T03 * T4;
T05 = T04 * T5;
T06 = T05 * T6;
simplify(T1\T06*[0; 0;DH(6,2); 1]-[0; 0; 0; 1])
th1 = 25*pi/180.0;
th2 = -45*pi/180.0;
th3 = 60*pi/180.0;
th4 = 0*pi/180.0; 
th5 = 15*pi/180.0; 
th6 = 30*pi/180.0;

fprintf("Joint positions\n")
fprintf('t1  = %f, t2  = %f, t3  = %f, t4  = %f, t5  = %f, t6  = %f\n', rad2deg(th1), rad2deg(th2), rad2deg(th3), rad2deg(th4), rad2deg(th5), rad2deg(th6));
h02 = vpa(subs(T02,[t1,t2],[th1,th2]));
h03 = vpa(subs(T03,[t1,t2,t3],[th1,th2,th3]));
h04 = vpa(subs(T04,[t1,t2,t3,t4],[th1,th2,th3,th4]));
h05 = vpa(subs(T05,[t1,t2,t3,t4,t5],[th1,th2,th3,th4,th5]));
h06 = vpa(subs(T06,[t1,t2,t3,t4,t5,t6],[th1,th2,th3,th4,th5,th6]));
fprintf('End_effector_position = %g %g %g\n', h06(1,4),h06(2,4),h06(3,4));

%% inverse kinematics start
desired_pose = h06;
r6b = desired_pose(1:3,1:3);
p6b = desired_pose(1:3,4);
q = rotm2quat(double(r6b));
fprintf('End_effector_quaternion = %g %g %g %g\n', q(1), q(2), q(3), q(4));
d1 = DH(1,2);
a2 = DH(2,3);
a3 = DH(3,3);
d4 = DH(4,2);
d5 = DH(5,2);
d6 = DH(6,2);

p_05 = desired_pose*[0; 0;-d6; 1]-[0; 0; 0; 1];
fprintf('wrist_position = %g %g %g\n', p_05(1),p_05(2),p_05(3));

%% calculating theta1 
a1 = atan2(p_05(2),p_05(1));
dxy = sqrt(p_05(1)^2 + p_05(2)^2);

beta_1 = acos(DH(4,2)/dxy);
% beta_2 = acos(DH(4,2)/dxy);
theta_1 = simplify(a1 + beta_1 - pi/2.0);

theta_1 = simplify(theta_1);
% fprintf('theta 1 = %f\n', rad2deg(theta_1));
%% calculating theta5
T1 = vpa( subs(Telem, [theta d a alpha], DH(1,:)) );
% T16 = simplify(T1\T06);
% T61 = simplify(inv(T16));

T01 = vpa(subs(T1,t1,theta_1));
T16 = simplify(T01\h06);
t16z = simplify(T16(3,4));
theta_5 = acos((t16z-d4)/d6); 
 
theta_5 = simplify(real(theta_5));
% fprintf('theta 5 = %f\n', rad2deg(theta_5));

%% calculating theta6
t16 = T01\h06;
t61 = simplify(inv(t16));

zx = t61(1,3);
zy = t61(2,3);
theta_6 = atan2(-zy/sin(theta_5),zx/sin(theta_5));
theta_6 = simplify(real(theta_6));
% fprintf('theta 6 = %f\n', rad2deg(theta_6));

%% calculating for Theta 3
t14 = simplify(t16/T6/T5);
t14 = vpa(subs(t14,[t1,t5,t6],[theta_1,theta_5,theta_6]));

p13 = t14*[0; -d4; 0; 1]-[0; 0; 0; 1];

dst = sqrt(p13(1)^2 + p13(2)^2 + p13(3)^2);
theta_3 = acos((dst^2 - a2^2 - a3^2)/(2*a2*a3)) ;
theta_3 = simplify(real(theta_3));
% fprintf('theta 3 = %f\n', rad2deg(theta_3));

%% calculating theta 2
theta_2 = -atan2(p13(2),-p13(1)) + asin((a3*sin(theta_3))/dst);
theta_2 = simplify(real(theta_2));
% fprintf('theta 2 = %f\n', rad2deg(theta_2));

%% calculating theta 4
t16 = T1\h06;
t26 = T2\t16;
t36 = T3\t26;
t34 = t36/T6/T5;

t34 = vpa(subs(t34,[t1,t2,t3,t5,t6],[theta_1,theta_2,theta_3,theta_5,theta_6]));
xx = simplify(t34(1,1));
xy = simplify(t34(2,1));
theta_4 = atan2(xy,xx);

theta_4 = simplify(real(theta_4));
% fprintf('theta 4 = %f\n', rad2deg(theta_4));

%% Results
fprintf('t1  = %f, t2  = %f, t3  = %f, t4  = %f, t5  = %f, t6  = %f\n', rad2deg(theta_1), rad2deg(theta_2), rad2deg(theta_3), rad2deg(theta_4), rad2deg(theta_5), rad2deg(theta_6));
