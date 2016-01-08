function [] = script()
	width = 320;
	height = 240;
    
    for i = 0:20
        joints = importdata(strcat('data/joints', int2str(i), '.dat'));
        plot3d(joints(:, 1:3), [-1000 1000 -1000 1000 1000 3000], false);
    end 
	%load data/label1.dat;
	%load data/depth1.dat;
    %load data/topJoints2.dat;
    %label = label1;
    %depth = depth1;
	%topJoints = topJoints2;

	%label = reshape(label, [width, height])';
	%label = mat2gray(label);
	%plot2d(label, joints(:, 4:5));

	%depth = reshape(depth, [width, height])';
	%depth = mat2gray(depth);
	%plot2d(depth, joints(:, 4:5));
    
    %plot3d(topJoints, [-1000 1000 -1000 1000 0 2000]);
    %plotProj(joints(:, 1:2));
    %plotProj(topJoints(:, 1:2));
    
    %bx = getParam(joints(:, 1), joints(:, 4));
    %by = getParam(joints(:, 2), joints(:, 5));
    %[topJoints(:,4), topJoints(:,5)] = world2depth(bx, by, topJoints(:,1), topJoints(:,2))
    %plotProj([topJoints(:,4), topJoints(:,5)], [0 320 0 240]);
end

function [xd, yd] = world2depth(bx, by, xw, yw)
    xd = bx(1) + bx(2)*xw;
    yd = by(1) + by(2)*yw;
end


function [b] = getParam(x, y)
    b = [ones(length(x), 1) x]\y;
end

function [] = plot2d(img, mat)
    figure; 
    imshow(img);
    hold on;
    scatter(mat(:, 1), mat(:, 2));
end

function [] = plotProj(mat, axisMat)
	figure;
	% JOINT_HEAD
	% JOINT_NECK 	
	% JOINT_LEFT_SHOULDER 	
	% JOINT_RIGHT_SHOULDER 	
	% JOINT_LEFT_ELBOW 	
	% JOINT_RIGHT_ELBOW 	
	% JOINT_LEFT_HAND 	
	% JOINT_RIGHT_HAND 	
	% JOINT_TORSO 	
	% JOINT_LEFT_HIP 	
	% JOINT_RIGHT_HIP 	
	% JOINT_LEFT_KNEE 	
	% JOINT_RIGHT_KNEE 	
	% JOINT_LEFT_FOOT 	
	% JOINT_RIGHT_FOOT 

	pts = [mat(1,:); mat(2,:)];
	plot(pts(:,1), pts(:,2));
	hold on;

	pts = [mat(2,:); mat(3,:)];
	plot(pts(:,1), pts(:,2));
	hold on;

	pts = [mat(2,:); mat(4,:)];
	plot(pts(:,1), pts(:,2));
	hold on;

	pts = [mat(5,:); mat(3,:)];
	plot(pts(:,1), pts(:,2));
	hold on;

	pts = [mat(6,:); mat(4,:)];
	plot(pts(:,1), pts(:,2));
	hold on;

	pts = [mat(5,:); mat(7,:)];
	plot(pts(:,1), pts(:,2));
	hold on;

	pts = [mat(6,:); mat(8,:)];
	plot(pts(:,1), pts(:,2));
	hold on;

	pts = [mat(9,:); mat(3,:)];
	plot(pts(:,1), pts(:,2));
	hold on;

	pts = [mat(9,:); mat(4,:)];
	plot(pts(:,1), pts(:,2));
	hold on;

	pts = [mat(9,:); mat(10,:)];
	plot(pts(:,1), pts(:,2));
	hold on;

	pts = [mat(9,:); mat(11,:)];
	plot(pts(:,1), pts(:,2));
	hold on;

	pts = [mat(10,:); mat(11,:)];
	plot(pts(:,1), pts(:,2));
	hold on;

	pts = [mat(12,:); mat(10,:)];
	plot(pts(:,1), pts(:,2));
	hold on;

	pts = [mat(13,:); mat(11,:)];
	plot(pts(:,1), pts(:,2));
	hold on;

	pts = [mat(12,:); mat(14,:)];
	plot(pts(:,1), pts(:,2));
	hold on;

	pts = [mat(13,:); mat(15,:)];
	plot(pts(:,1), pts(:,2));
	hold on;

	xlabel('x');
	ylabel('y');
    axis equal;
    axis(axisMat);
end

function [] = plot3d(mat, axisMat, fill)
    figure;
	% JOINT_HEAD
	% JOINT_NECK 	
	% JOINT_LEFT_SHOULDER 	
	% JOINT_RIGHT_SHOULDER 	
	% JOINT_LEFT_ELBOW 	
	% JOINT_RIGHT_ELBOW 	
	% JOINT_LEFT_HAND 	
	% JOINT_RIGHT_HAND 	
	% JOINT_TORSO 	
	% JOINT_LEFT_HIP 	
	% JOINT_RIGHT_HIP 	
	% JOINT_LEFT_KNEE 	
	% JOINT_RIGHT_KNEE 	
	% JOINT_LEFT_FOOT 	
	% JOINT_RIGHT_FOOT 

	pts = [mat(1,:); mat(2,:)]; % JOINT_HEAD, JOINT_NECK 
	plot3(pts(:,1), pts(:,2), pts(:,3));
    if fill
        center = (mat(1,:) + mat(2,:)) ./ 2;
        radius = pdist(pts, 'euclidean') / 2;
        [x,y,z] = sphere; 
        surf(x*radius+center(1),y*radius+center(2),z*radius+center(3));
    end
	hold on;

	pts = [mat(2,:); mat(3,:)];
	plot3(pts(:,1), pts(:,2), pts(:,3));
	hold on;

	pts = [mat(2,:); mat(4,:)];
	plot3(pts(:,1), pts(:,2), pts(:,3));
	hold on;

	pts = [mat(5,:); mat(3,:)]; % JOINT_LEFT_SHOULDER, JOINT_LEFT_ELBOW
    if fill 
        plot3(pts(:,1), pts(:,2), pts(:,3), 'LineWidth', 10);
    else 
        plot3(pts(:,1), pts(:,2), pts(:,3));
    end
	hold on;

	pts = [mat(6,:); mat(4,:)]; % JOINT_RIGHT_SHOULDER, JOINT_RIGHT_ELBOW
    if fill 
	    plot3(pts(:,1), pts(:,2), pts(:,3), 'LineWidth', 10);
    else
        plot3(pts(:,1), pts(:,2), pts(:,3));
    end
	hold on;

	pts = [mat(5,:); mat(7,:)];
    if fill
	    plot3(pts(:,1), pts(:,2), pts(:,3), 'LineWidth', 10);
    else
        plot3(pts(:,1), pts(:,2), pts(:,3));
    end
	hold on;

	pts = [mat(6,:); mat(8,:)];
    if fill
	    plot3(pts(:,1), pts(:,2), pts(:,3), 'LineWidth', 10);
    else
        plot3(pts(:,1), pts(:,2), pts(:,3));
    end
	hold on;

	pts = [mat(9,:); mat(3,:)];
	plot3(pts(:,1), pts(:,2), pts(:,3));
	hold on;

	pts = [mat(9,:); mat(4,:)];
	plot3(pts(:,1), pts(:,2), pts(:,3));
	hold on;

	pts = [mat(9,:); mat(10,:)];
	plot3(pts(:,1), pts(:,2), pts(:,3));
	hold on;

	pts = [mat(9,:); mat(11,:)];
	plot3(pts(:,1), pts(:,2), pts(:,3));
	hold on;

	pts = [mat(10,:); mat(11,:)];
	plot3(pts(:,1), pts(:,2), pts(:,3));
	hold on;

	pts = [mat(12,:); mat(10,:)];
    if fill
        plot3(pts(:,1), pts(:,2), pts(:,3), 'LineWidth', 20);
    else
        plot3(pts(:,1), pts(:,2), pts(:,3));
    end
	hold on;

	pts = [mat(13,:); mat(11,:)];
    if fill
        plot3(pts(:,1), pts(:,2), pts(:,3), 'LineWidth', 20);
    else
        plot3(pts(:,1), pts(:,2), pts(:,3));
    end
	hold on;

	pts = [mat(12,:); mat(14,:)];
    if fill
        plot3(pts(:,1), pts(:,2), pts(:,3), 'LineWidth', 20);
    else
        plot3(pts(:,1), pts(:,2), pts(:,3));
    end
	hold on;

	pts = [mat(13,:); mat(15,:)];
    if fill
	    plot3(pts(:,1), pts(:,2), pts(:,3), 'LineWidth', 20);
    else
        plot3(pts(:,1), pts(:,2), pts(:,3));
    end
	hold on;

    if fill
        mat(13,:); mat(15,:)
        pt1 = (mat(3,:) + mat(4,:)) ./ 2;
        pt2 = (mat(3,:) + pt1) ./ 2;
        pt3 = (mat(4,:) + pt1) ./ 2;
        pt4 = (mat(10,:) + mat(11,:)) ./ 2;
        pt5 = (mat(10,:) + pt4) ./ 2;
        pt6 = (mat(11,:) + pt4) ./ 2;
        width = pdist([pt1; pt3], 'euclidean') / 2;

        pts = [pt2; pt5];
        plot3(pts(:,1), pts(:,2), pts(:,3), 'LineWidth', width);  
        pts = [pt3; pt6];
        plot3(pts(:,1), pts(:,2), pts(:,3), 'LineWidth', width);
    end
    
    xlabel('x');
	ylabel('y');
	zlabel('z');
    axis(axisMat);
    view([0,0,1]);
end
