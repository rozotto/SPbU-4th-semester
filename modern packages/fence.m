function m = fence(varargin)
  if nargin == 1
    rows = varargin{1};
    cols = rows;
  else
    rows = varargin{1};
    cols = varargin{2};
  end
  m = zeros(rows, cols, 'logical');
  m(:, 1) = 1;
  for j = 2:cols
    m(:, j) = mod(j, 2);
  end
end
