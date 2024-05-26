function m = checkerboard(varargin)
  if nargin == 1
    rows = varargin{1};
    cols = rows;
  else
    rows = varargin{1};
    cols = varargin{2};
  end
  m = zeros(rows, cols, 'logical');
  for i = 1:rows
    for j = 1:cols
      if mod(i+j, 2) == 0
        m(i, j) = 1;
      end
    end
  end
endfunction
