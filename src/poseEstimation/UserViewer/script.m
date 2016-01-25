function [] = script()
%    visualizeOne();
%     for i = 0:47
%         jointsSide = importdata(strcat('data/joints-side', int2str(i), '.dat'));
%         figure;
%         plot3d(jointsSide(:, 1:3), [0 2000 -1000 1000 2000 4000], false);
%         jointsTop = importdata(strcat('data/joints-top', int2str(i), '.dat'));
%         figure;
%         plot3d(jointsTop(:, 1:3), [-2000 0 -1000 1000 0 2000], false);
%         pause;
%         clf;
%         %plotProj(joints(:, 1:2));
%     end 
    
% 	label = reshape(label, [width, height])';
% 	label = mat2gray(label);
% 	plot2d(label, joints(:, 4:5));
%
    %visualizeOne();
    joints2pixel();
end

function [] = joints2pixel()
	width = 320;
	height = 240;
    
    depth = importdata(strcat('data/depth', int2str(1), '.dat'));
 	depth = reshape(depth, [width, height])';
    jointsTop = importdata(strcat('data/joints-top', int2str(0), '.dat'));
    
    figure;
    imshow(uint8(depth./12));
    
    X = [jointsTop(:, 4:5) jointsTop(:, 3)/4.74];
    Y = zeros(width*height, 3);
    
    for i = 1:width
        tmp1 = (repmat(i, 1, height))';
        tmp2 = (1:height)';
        Y((i-1)*height+1:i*height, 1) = tmp1;
        Y((i-1)*height+1:i*height, 2) = tmp2;
        Y((i-1)*height+1:i*height, 3) = depth(:, i)/4.74;
    end
    
    IDX = knnsearch(X, Y);
    %size(X)
    %size(Y)
    %size(IDX)
    
    img = zeros(height, width, 3, 'uint8');
    palette1 = [
                0, 0, 255;...
                0, 255, 0;...
                255, 0, 0;...
                0, 255, 255;...
                255, 0, 255;...
                255, 255, 0;...
                0, 0, 128;...
                0, 128, 0;...
                128, 0, 0;...
                0, 128, 128;...
                128, 0, 128;...
                128, 128, 0;...
                0, 0, 0;...
                255, 255, 255;...
                128, 128, 128
                ];
    palette2 = palette1 ./ 255;
    
    for i = 1:width
        for j = 1:height
            img(j, i, :) = palette1(IDX((i-1)*height+j), :);
        end
    end 

    figure;
    imshow(img);
    
    figure;
    for i = 1:15
        plot([1, i], [2, i], 'Color', palette2(i, :));
        hold on;
    end
    legend('head', 'neck', 'left shoulder', 'right shoulder', ...
        'left elbow', 'right elbow', 'left hand', 'right hand', ...
        'turso', 'left hip', 'right hip', 'left knee', 'right knee', ...
        'left foot', 'right foot');
end

function[] = visualizeOne()
    depth = importdata(strcat('data/depth', int2str(0), '.dat'));
 	depth = reshape(depth, [320, 240])';
 	depth = mat2gray(depth);
    imshow(depth);

    jointsSide = importdata(strcat('data/joints-side', int2str(0), '.dat'));
    plot3d(jointsSide(:, 1:3), [-1000 1000 -1000 1000 2000 4000], false);
    plotProj(jointsSide(:, 4:5), [0 320 0 240]);
    
    jointsTop = importdata(strcat('data/joints-top', int2str(0), '.dat'));
    plot3d(jointsTop(:, 1:3), [-1000 1000 -1000 1000 1000 3000], false);
    plotProj(jointsTop(:, 4:5), [0 320 0 240]);
end

function [] = plot2d()
	width = 320;
	height = 240;
    depth = importdata(strcat('data/depth', int2str(1), '.dat'));
 	depth = reshape(depth, [width, height])';
 	depth = mat2gray(depth);
    jointsTop = importdata(strcat('data/joints-top', int2str(0), '.dat'));
    img = depth;
    mat = jointsTop(:, 4:5);
    
    figure; 
    imshow(img);
    hold on;
    scatter(mat(:, 1), mat(:, 2));
    
    pts = [mat(1,:); mat(2,:)];
	plot(pts(:,1), pts(:,2), 'LineWidth', 5);
	hold on;

	pts = [mat(2,:); mat(3,:)];
	plot(pts(:,1), pts(:,2), 'LineWidth', 4);
	hold on;

	pts = [mat(2,:); mat(4,:)];
	plot(pts(:,1), pts(:,2), 'LineWidth', 4);
	hold on;

	pts = [mat(5,:); mat(3,:)];
	plot(pts(:,1), pts(:,2), 'LineWidth', 4);
	hold on;

	pts = [mat(6,:); mat(4,:)];
	plot(pts(:,1), pts(:,2), 'LineWidth', 4);
	hold on;

	pts = [mat(5,:); mat(7,:)];
	plot(pts(:,1), pts(:,2), 'LineWidth', 4);
	hold on;

	pts = [mat(6,:); mat(8,:)];
	plot(pts(:,1), pts(:,2), 'LineWidth', 4);
	hold on;

	pts = [mat(9,:); mat(3,:)];
	plot(pts(:,1), pts(:,2), 'LineWidth', 3);
	hold on;

	pts = [mat(9,:); mat(4,:)];
	plot(pts(:,1), pts(:,2), 'LineWidth', 3);
	hold on;

	pts = [mat(9,:); mat(10,:)];
	plot(pts(:,1), pts(:,2), 'LineWidth', 3);
	hold on;

	pts = [mat(9,:); mat(11,:)];
	plot(pts(:,1), pts(:,2), 'LineWidth', 3);
	hold on;

	pts = [mat(10,:); mat(11,:)];
	plot(pts(:,1), pts(:,2), 'LineWidth', 2);
	hold on;

	pts = [mat(12,:); mat(10,:)];
	plot(pts(:,1), pts(:,2), 'LineWidth', 2);
	hold on;

	pts = [mat(13,:); mat(11,:)];
	plot(pts(:,1), pts(:,2), 'LineWidth', 2);
	hold on;

	pts = [mat(12,:); mat(14,:)];
	plot(pts(:,1), pts(:,2), 'LineWidth', 2);
	hold on;

	pts = [mat(13,:); mat(15,:)];
	plot(pts(:,1), pts(:,2), 'LineWidth', 2);
	hold on;
end

function [] = plot3d(mat, limits, fill)
    figure;

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
    axis(limits);
    view([0,0,1]);
end

function [] = plotProj(mat, limits)
	figure;

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
    axis(limits);
end
