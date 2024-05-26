function baguett(l, d, a, n)
  angle = a * pi / 180;
  depth = l / n;

  figure('Position', [100 100 800 400]);
  subplot(1, 2, 1);

  [x, y, z] = cylinder(d / 2, 50);
  colormap gray;
  z = z * l;
  surf(x, y, z);
  axis equal;

  subplot(1, 2, 2);

  for i = 1:2:n
    start = (i - 1) * depth;
    stop = start + depth;

    x1 = d / 2 * cos(angle);
    y1 = d / 2 * sin(angle);
    x2 = d / 2 * cos(angle + pi);
    y2 = d / 2 * sin(angle + pi);

    [x, y, z] = cylinder(d / 2, 50);
    z = z * depth;
    surf(x, y, z + start);
    hold on;

    plot3([x1, x2], [y1, y2], [start, start], 'k-', 'LineWidth', 4);
    plot3([x1, x2], [y1, y2], [stop, stop], 'k-', 'LineWidth', 4);
  end

  axis equal;
end
