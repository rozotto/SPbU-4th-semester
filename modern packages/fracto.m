function fracto(coefficients, left_top, right_bottom, num_points_x, num_points_y)
  dp = polyder(coefficients);
  root = roots(coefficients);
  x_range = linspace(left_top(1), right_bottom(1), num_points_x);
  y_range = linspace(right_bottom(2), left_top(2), num_points_y);
  [X, Y] = meshgrid(x_range, y_range);
  Z = X + 1i*Y;

  for iter = 1:100
    Z = Z - polyval(coefficients, Z) ./ polyval(dp, Z);
  end

  colors = spring(length(coefficients) - 1);
  tmp = Z;

  for k = 1:length(root)
    distances = abs(Z - root(k));
    is_close = distances < 0.001;
    tmp(is_close == 1) =  100 * rgb2gray(colors(k, :));
  end

   image(tmp);
endfunction
