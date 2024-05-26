function hypersurf(A)
    if ~ishyper(A)
        disp('Не однополостный гиперболоид или гиперболический параболоид.');
        return;
    end

    [X, Y] = meshgrid(linspace(-5, 5, 100), linspace(-5, 5, 100));
    Z = (-A(1, 1) * X.^2 - A(2, 2) * Y.^2 - A(3, 3)) / A(4, 4);

    surf(X, Y, Z);
    xlabel('X');
    ylabel('Y');
    zlabel('Z');
    axis equal;
    grid on;
end
