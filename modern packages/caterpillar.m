function caterpillar(x, y, n)
    if length(x) < n || length(y) < n
        disp('Недостаточно точек для построения гусеницы');
        return;
    end

    for i = 1:n
        plot(x(1:i), y(1:i), '-o');
        axis equal;
        xlabel('X');
        ylabel('Y');
    end
end
