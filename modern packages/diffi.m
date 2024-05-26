function deriv = diffi(f, ts, h=1e-8)
    num_points = length(ts);
    deriv = zeros(2, num_points);
    
    for i = 1:num_points
        t = ts(i);
        deriv(:, i) = (f(t + h) - f(t)) / h;
    end
end
