function m = mirrorm (a, n)
    [rows,cols] = size(a);
    a_ud = flipud(a);
    a_lr = fliplr(a);
    a_c = flipud(a_lr).'
    a_t = a.'
    A = [a_c(cols - n + 1:end, rows - n + 1:end), a_ud(rows-n+1:end, :), a_t(cols - n + 1:end, 1:n);
    a_lr(:, cols - n + 1:end), a, a_lr(:, 1:n);
    a_t(1:n, rows - n + 1:end), a_ud(1:n, :), a_c(1:n, 1:n)];
    m = uint32(A);
endfunction
