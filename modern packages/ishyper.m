function result = ishyper(matrix)
    detA = det(matrix);
    if or(detA > 0, detA < 0)
        result = true;
    else
        result = false;
    end
end
