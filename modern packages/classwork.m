#{
x = 0:pi/100:4*pi;
y = x;
[X,Y] = meshgrid(x, y);
z = 3*sin(X) + cos(Y);
h = surf(z);
axis tight;
shading interp;
colormap(ocean);
for k = 0:pi/100:2*pi
  z = (sin(X) + cos(Y)) .* sin(k);
  set(h, 'Zdata', z);
  drawnow
endfor
#}

#{
n = 100;
t1 = pi*(-n:5:n)/n;
t2 = (pi/2)*(-n:5:n)'/n;
X = cos(t2)*cos(t1);
Y = cos(t2)*sin(t1);
E = ones(size(t1));
Z = sin(t2)*E;
plot3(X, Y, Z, 'b'), grid on
#}

#{
x = -10:0.5:10;
y = -10:0.5:10;
[X, Y] = meshgrid(x,y);
Z = sin(sqrt(X.^2+Y.^2))./sqrt(X.^2+Y.^2);
figure
mesh(X, Y, Z)
#}

#{
u = 0:0.1:2*pi + 1;
v = -1:0.1:1;
[U,V] = meshgrid(u,v);
X = (1 + (V/2) .* cos(U/2)).*cos(U);
Y = (1 + (V/2) .* cos(U/2)).*sin(U);
Z = (V/2 .* sin(U/2));
surf(X,Y,Z);
shading interp;
#}

p = @(z) z.^3 - 1;
dp = @(z) 3*z.^2;
exact_roots = roots([1, 0, 0, -1]);

x_range = linspace(-2, 2, 500);
y_range = linspace(-2, 2, 500);
[X, Y] = meshgrid(x_range, y_range);
Z = X + 1i*Y;

for iter = 1:41
  Z = Z - p(Z)./dp(Z);
end

hold on;
colors = ['r', 'g', 'b'];

for k = 1:length(exact_roots)
  distances = abs(Z - exact_roots(k));
  is_close = distances < 0.001;
  plot(X(is_close), Y(is_close), '.', 'Color', colors(k), 'MarkerSize', 1);
end

hold off;

