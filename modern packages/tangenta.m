function tangenta(f, dmin, dmax, n)
  ts = linspace(dmin, dmax, 2*n + 1);
  points = zeros(2, length(ts));

  for i = 1:length(ts)
    points(:, i) = f(ts(i));
  end

  mid_idx = ceil(length(ts) / 2);
  t_mid = ts(mid_idx);
  p_mid = f(t_mid);

  derivatives = diffi(f, ts, 1e-8);
  tangent_vector = derivatives(:, mid_idx);

  t_length = 1;
  tangent_start = p_mid - t_length * tangent_vector;
  tangent_end = p_mid + t_length * tangent_vector;

  plot(points(1, :), points(2, :), 'b', 'LineWidth', 2);
  hold on;
  plot([tangent_start(1), tangent_end(1)], [tangent_start(2), tangent_end(2)], 'r--', 'LineWidth', 2);
  plot(p_mid(1), p_mid(2), 'go', 'MarkerSize', 10);

  xlabel('x');
  ylabel('y');
  title('Кривая и касательная');
  legend('Кривая', 'Касательная', 'Средняя точка');

end
