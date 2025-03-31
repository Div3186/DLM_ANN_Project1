
def model_summary_to_df(model):
    summary_list = []
    model.summary(print_fn=lambda x: summary_list.append(x))
    return "\n".join(summary_list)
