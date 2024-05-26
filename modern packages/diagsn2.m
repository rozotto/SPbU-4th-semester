function m = diagsn(varargin)
  if nargin == 1
    rows = varargin{1};
    cols = rows;
  else
    rows = varargin{1};
    cols = varargin{2};
  end
  m = zeros(rows, cols, 'uint32');
  num = 1;
  for i = 1:(rows+cols-1)
    if mod(i, 2) == 1
      if i <= cols
        start_row = 1;
        start_col = i;
      else
        start_row = i - cols + 1;
        start_col = cols;
      end
      while start_row <= rows && start_col >= 1
        m(start_row, start_col) = num;
        num = num + 1;
        start_row = start_row + 1;
        start_col = start_col - 1;
      end
    else
      if i <= rows
        start_row = i;
        start_col = 1;
      else
        start_row = rows;
        start_col = i - rows + 1;
      end
      while start_row >= 1 && start_col <= cols
        m(start_row, start_col) = num;
        num = num + 1;
        start_row = start_row - 1;
        start_col = start_col + 1;
      end
    end
  end
end
