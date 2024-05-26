function caterpool(pos, spee, bl, tr, n)
  pos = pos;
  vel = spee;
  leaf_width = tr(1) - bl(1);
  leaf_height = tr(2) - bl(2);

  figure;
  hold on;
  rectangle('Position', [bl, leaf_width, leaf_height], 'FaceColor', 'w', 'EdgeColor', 'r');

  for i = 1:n
    center_x = pos(1);
    center_y = pos(2);
    radius = 1;

    theta = linspace(0, 2*pi, 50);
    circle_x = center_x + radius * cos(theta);
    circle_y = center_y + radius * sin(theta);
    plot(circle_x, circle_y, '--');

    pos = pos + vel;

    if pos(1) - radius < bl(1)
      vel(1) = -vel(1);
      pos(1) = bl(1) + radius;
    elseif pos(1) + radius > tr(1)
      vel(1) = -vel(1);
      pos(1) = tr(1) - radius;
    end

    if pos(2) - radius < bl(2)
      vel(2) = -vel(2);
      pos(2) = bl(2) + radius;
    elseif pos(2) + radius > tr(2)
      vel(2) = -vel(2);
      pos(2) = tr(2) - radius;
    end
  end

  axis equal;
  xlabel('X');
  ylabel('Y');
end
