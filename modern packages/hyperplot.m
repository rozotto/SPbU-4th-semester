function hyperplot(A)
    if ~ishyper(A)
        disp('Не однополостный гиперболоид или гиперболический параболоид.');
        return;
    end

    [V, D] = eig(A(1:3, 1:3));
    if any(imag(D(:)))
        disp('Комплексные собственные значения.');
        return;
    end

    t = linspace(-10, 10, 100);
    figure;
    hold on;
    for i = 1:3
        plot3(V(1, i) * t, V(2, i) * t, V(3, i) * t, 'r', 'LineWidth', 2);
    end
    xlabel('X');
    ylabel('Y');
    zlabel('Z');
    axis equal;
    grid on;
    hold off;
end
