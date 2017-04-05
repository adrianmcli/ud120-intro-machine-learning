#!/usr/bin/python


def outlierCleaner(predictions, ages, net_worths):
    """
        Clean away the 10% of points that have the largest
        residual errors (difference between the prediction
        and the actual net worth).

        Return a list of tuples named cleaned_data where 
        each tuple is of the form (age, net_worth, error).
    """
    
    cleaned_data = []

    # predictions: list of predicted targets (predicted_net_worth) that come from your regression
    # ages: list of ages in the training set
    # net_worths: the actual value of the net worths in the training set
    num_data_points = len(predictions)

    for i in range(num_data_points):
        # Extract values
        age = ages[i][0]
        prediction = predictions[i][0]
        net_worth = net_worths[i][0]

        # Calculate error and create tuple
        error = abs(prediction - net_worth)
        result_tuple = (age, net_worth, error)

        # Append tuple to list
        cleaned_data.append(result_tuple)

    # Sort by the error and take only the first 90%
    cleaned_data = sorted(cleaned_data, key=lambda x: x[2])
    cleaned_data = cleaned_data[:int(num_data_points * 0.9)]

    return cleaned_data

