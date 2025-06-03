import os

# concept evaluation prompt
def concept_prompt_process(concept, type_, text):
    feature_details_dict = {
        "animal": f"Evaluate the accuracy, completeness, and logical placement of the {concept}'s features. Focus on facial details, limbs, and skin texture to ensure they align with the natural or expected representation of the {concept}.",
        "pet": f"Evaluate the accuracy, completeness, and logical placement of the {concept}'s features. Focus on facial details, limbs, and skin texture to ensure they align with the natural or expected representation of the {concept}.",
        "person": f"Evaluate the accuracy, completeness, and logical placement of the {concept}'s features. Focus on facial details, limbs, and skin texture to ensure they align with the natural or expected representation of the {concept}.",
        "plant": f"Evaluate the accuracy, completeness, and logical placement of the {concept}'s features. Focus on facial details, limbs, and skin texture to ensure they align with the natural or expected representation of the {concept}.",
        "hat": f"Evaluate the accuracy, completeness, and logical placement of the {concept}'s features. Focus on functional components, material details, and decorative elements to ensure they align with the natural or expected representation of the {concept}.",
        "bag": f"Evaluate the accuracy, completeness, and logical placement of the {concept}'s features. Focus on functional components, material details, and decorative elements to ensure they align with the natural or expected representation of the {concept}.",
        "car": f"Evaluate the accuracy, completeness, and logical placement of the {concept}'s features. Focus on functional components, material details, and decorative elements to ensure they align with the natural or expected representation of the {concept}.",
        "clothes": f"Evaluate the accuracy, completeness, and logical placement of the {concept}'s features. Focus on functional components, material details, and decorative elements to ensure they align with the natural or expected representation of the {concept}.",
        "food": f"Evaluate the accuracy, completeness, and logical placement of the {concept}'s features. Focus on functional components, material details, and decorative elements to ensure they align with the natural or expected representation of the {concept}.",
        "music_instrument": f"Evaluate the accuracy, completeness, and logical placement of the {concept}'s features. Focus on functional components, material details, and decorative elements to ensure they align with the natural or expected representation of the {concept}.",
        "electronic": f"Evaluate the accuracy, completeness, and logical placement of the {concept}'s features. Focus on functional components, material details, and decorative elements to ensure they align with the natural or expected representation of the {concept}.",
        "sport_equipment": f"Evaluate the accuracy, completeness, and logical placement of the {concept}'s features. Focus on functional components, material details, and decorative elements to ensure they align with the natural or expected representation of the {concept}.",
        "other": f"Evaluate the accuracy, completeness, and logical placement of the {concept}'s features. Focus on functional components, material details, and decorative elements to ensure they align with the natural or expected representation of the {concept}.",
        "celestial": f"Examine the {concept}’s architectural details, unique decorations, and structural features, ensuring key elements and iconic details are precisely represented with correct proportions, symmetry, spatial layout and accurately reflect the original design.",
        "landmark": f"Examine the {concept}’s architectural details, unique decorations, and structural features, ensuring key elements and iconic details are precisely represented with correct proportions, symmetry, spatial layout and accurately reflect the original design.",
        "natural_landform": f"Examine the {concept}’s architectural details, unique decorations, and structural features, ensuring key elements and iconic details are precisely represented with correct proportions, symmetry, spatial layout and accurately reflect the original design.",
        "event": f"Evaluate the {concept}'s portrayal by examining the historical accuracy and representation of key figures, attire, and iconic scenes within the context of the event. Ensure that the visual narrative accurately depicts significant moments and the overall atmosphere, maintaining fidelity to documented historical accounts and cultural settings.",
        "music": f"Evaluate the accuracy, completeness, and logical placement of the {concept}'s features. Focus on functional components, material details, and decorative elements to ensure they align with the natural or expected representation of the {concept}.",
        "sport": f"Evaluate the accuracy, completeness, and logical placement of the {concept}'s features. Focus on functional components, material details, and decorative elements to ensure they align with the natural or expected representation of the {concept}.",
        "artifact": f"Evaluate the accuracy, completeness, and logical placement of the {concept}'s features. Focus on functional components, material details, and decorative elements to ensure they align with the natural or expected representation of the {concept}.",
        "location": f"Examine the {concept}’s architectural details, unique decorations, and structural features, ensuring key elements and iconic details are precisely represented with correct proportions, symmetry, spatial layout and accurately reflect the original design.",
    }
    feature_details = feature_details_dict[type_]
    concept_eva = f'''
    You are provided with a concept image and a generated image. First, learn the concept of {concept} from the concept image. Then evaluate the {concept} in the generated image based on your own knowledge of {concept} and the concept of {concept} you learned from the concept image.            
    Concept Evaluation:
    Evaluate whether the {concept} in the generated image matches the conceptual characteristics of the {concept} based on:
    1. Shape Accuracy: Focus on the overall outline and structure of the generated {concept}. Assess whether the overall silhouette, pose, and proportions align with the common shapes associated with the concept. 
    2. Color Accuracy: Assess whether the generated {concept}’s color scheme and lighting conditions align with the natural or expected hues, saturation, and brightness characteristic of the concept.
    3. Texture Representation: Evaluate the realism and clarity of the {concept}’s textures, ensuring authentic representation in key areas, free from blurriness, pixelation, or artificial effects, to uphold realistic integrity.
    4. Feature Details: {feature_details}

    Notice that: You should only consider the concept of {concept} in the generated image.
    Evaluation Criteria:
    Evaluate based on the criteria above. Score from 0 to 4 based on how many criteria are fully met:
    - 0: None met
    - 1: One met
    - 2: Two met
    - 3: Three met
    - 4: All met
    Explain your rating by providing a very brief core reason. Keep your explanation concise and to the point.
    Input:
    The generated image <image>. Caption of the generated image is {text}.
    The concept image <image>. Caption of the concept image is {concept}.

    Output Format(Notice that output must be without any bold formatting):
    Total Rating: Your Rating
    Shape Accuracy: 0/1 Reason:
    Color Accuracy: 0/1 Reason:
    Texture Representation: 0/1 Reason:
    Feature Details: 0/1 Reason:
    '''               
    return concept_eva

# task evaluation prompt
def task_prompt_process(line, bandf):
    if not bandf:
        if line["task"] == "action":
            for concept, type_, concept_image in zip(line["concept"], line["type"], line["reference_image"]):
                task = "action"
                text = line["sentence"]
                # image_concept_path = os.path.join(concept_base, concept_image.split("/")[-2], concept_image.split("/")[-1])
                # result_image = line["result_image"]
                # image_generated_path = os.path.join(generate_base, result_image.split("/")[-2], result_image.split("/")[-1])
                addition = line["addition"][0]
                # print(image_concept_path, image_generated_path)
                concept_eva = f'''
                You are provided with a generated image, its corresponding caption, a knowledge concept {type_} {concept}, and an action: {addition}. Your task is: 
                1. Confirm Concept Presence: Check whether {concept} is present in the image. If it is not present, the Task Score should be 0.
                2. Confirm Action: If {concept} is present, determine whether the {type_} {concept} in the generated image is either dynamically performing the action of {addition} or is being dynamically acted upon. If either of these conditions is true, the Task Score should be 1; if neither is true, the Task Score should be 0.
                Evaluation Order: Please first confirm the presence of the concept, then verify the action of knowledge concept.

                Output Format (Note: the output must be without any bold formatting):
                Task Score:
                Reason:
                '''               
                return concept_eva
        elif line["task"] == "attribute":
            for concept, type_, concept_image in zip(line["concept"], line["type"], line["reference_image"]):
                task = "attribute"
                text = line["sentence"]
                #image_concept_path = os.path.join(concept_base, concept_image.split("/")[-2], concept_image.split("/")[-1])
                #result_image = line["result_image"]
                #image_generated_path = os.path.join(generate_base, result_image.split("/")[-2], result_image.split("/")[-1])
                addition = line["addition"][0]
                #print(image_concept_path, image_generated_path)
                concept_eva = f'''
                You will be provided with a generated image, a knowledge concept {type_} {concept}, and an attribute related to concept: {addition}. Your task is:
                1. Confirm Concept Presence: Check whether {concept} is present in the image. If it is not present, the Task Score should be 0.
                2. Confirm Attribute Change: If {concept} is present, assess whether the knowledge concept exhibits the attribute {addition}. If it does, the Task Score should be 1; otherwise, the Task Score should be 0.
                Evaluation Order: Please first confirm the presence of the concept, then verify the knowledge concept attribute.
                Output Format (Note: do not use any bold formatting):
                Task Score:
                Reason:
                '''               
                return concept_eva
        elif line["task"] == "background variation" or line["task"] == "variation":
            for concept, type_, concept_image in zip(line["concept"], line["type"], line["reference_image"]):
                text = line["sentence"]
                #image_concept_path = os.path.join(concept_base, concept_image.split("/")[-2], concept_image.split("/")[-1])
                #result_image = line["result_image"]
                #image_generated_path = os.path.join(generate_base, result_image.split("/")[-2], result_image.split("/")[-1])
                attribute = line["addition"][0]
                #print(image_concept_path, image_generated_path)
                concept_eva = f'''
                You will be provided with a generated image, a background knowledge concept {type_} {concept}, and an attribute related to background: {attribute}. Your task is:
                1. Confirm Concept Presence: Check whether {concept} is present in the image. If it is not present, the Task Score should be 0.
                2. Confirm Background Attribute: If {concept} is present, assess whether the related background exhibits the attribute {attribute}. If it does, the Task Score should be 1; otherwise, the Task Score should be 0.
                Evaluation Order: Please first confirm the presence of the concept, then verify its background attribute.

                Output Format (Note: do not use any bold formatting):
                Task Score:
                Reason:
                '''               
                return concept_eva
        elif line["task"] == "interaction":
            type_1 = line["type"][0]
            type_2 = line["type"][1]
            concept_1 = line["concept"][0]
            concept_2 = line["concept"][1]
            concept_image_1 = line["reference_image"][0]
            concept_image_2 = line["reference_image"][1]

            task = "interaction"
            text = line["sentence"]
            
            # image_concept_path_1 = os.path.join(concept_base, concept_image_1.split("/")[-2], concept_image_1.split("/")[-1])
            # image_concept_path_2 = os.path.join(concept_base, concept_image_2.split("/")[-2], concept_image_2.split("/")[-1])
            # result_image = line["result_image"]
            # image_generated_path = os.path.join(generate_base, result_image.split("/")[-2], result_image.split("/")[-1])
            addition = line["addition"][0]
            # print(image_concept_path_1)
            # print(image_concept_path_2)
            # print(image_generated_path)
            concept_eva = f'''
            You are provided with a generated image and two knowledge concepts: {type_1} {concept_1} and {type_2} {concept_2}. Additionally, an interaction action is specified: {addition}. Your task is to :
            1. Confirmation of Presence: determine whether both {concept_1} and {concept_2} are present in the generated image. If either is not present, the Task Score should be 0. If both are present:
            2. Interaction Verification: determine whether they are dynamically interacting through the specified interaction of {addition}. If this interaction is occurring, the Task Score should be 1; if not, the Task Score should be 0.
            Evaluation Order: Please first confirm the presence of the concepts, then proceed to verify their interaction.
            Input:
            1. The generated image: <image>.
            2. Concepts: {type_1} {concept_1}, {type_2} {concept_2}, Interaction: {addition}
            Output Format (Note: the output must be without any bold formatting):
            Task Score: 
            Reason: 
            '''               
            return concept_eva
        elif line["task"] == "size":
            type_1 = line["type"][0]
            type_2 = line["type"][1]
            concept_1 = line["concept"][0]
            concept_2 = line["concept"][1]
            concept_image_1 = line["reference_image"][0]
            concept_image_2 = line["reference_image"][1]

            task = "interaction"
            text = line["sentence"]
            
            # image_concept_path_1 = os.path.join(concept_base, concept_image_1.split("/")[-2], concept_image_1.split("/")[-1])
            # image_concept_path_2 = os.path.join(concept_base, concept_image_2.split("/")[-2], concept_image_2.split("/")[-1])
            # result_image = line["result_image"]
            # image_generated_path = os.path.join(generate_base, result_image.split("/")[-2], result_image.split("/")[-1])
            addition = line["addition"][0]
            # print(image_concept_path_1)
            # print(image_concept_path_2)
            # print(image_generated_path)
            concept_eva = f'''
            You need to analyze a generated image that contains two knowledge concepts: {type_1} {concept_1} (for example, "Husky") and {type_2} {concept_2} (for example, "Egyptian Cat"). The task details are as follows:
            1. Existence Check: First, confirm whether both concepts are present in the generated image. If either concept is missing, the Task Score should be 0.
            2. Size Restoration: If both concepts are present, assess whether their size proportions in the image match their real-world counterparts.
            3. Evaluation Criteria: Estimate the relative volume, height, or recognizable size ratio of each concept in the image to ensure they reflect the real-world size differences. If {concept_1} is indeed larger than {concept_2} as expected, the Task Score should be 1; otherwise, it should be 0.
            Input Information:

            The generated image: <image>.
            Concepts: {type_1} {concept_1} , {type_2} {concept_2}
            Output Format (Note: the output must be without any bold formatting):
            Task Score:
            Reason:
            '''               
            return concept_eva
        elif line["task"] == "differentiating":
            type_1 = line["type"][0]
            type_2 = line["type"][1]
            concept_1 = line["concept"][0].replace("_", " ")
            concept_2 = line["concept"][1].replace("_", " ")
            concept_image_1 = line["reference_image"][0]
            concept_image_2 = line["reference_image"][1]

            task = "interaction"
            text = line["sentence"]
            
            # image_concept_path_1 = os.path.join(concept_base, concept_image_1.split("/")[-2], concept_image_1.split("/")[-1])
            # image_concept_path_2 = os.path.join(concept_base, concept_image_2.split("/")[-2], concept_image_2.split("/")[-1])
            # result_image = line["result_image"]
            # image_generated_path = os.path.join(generate_base, result_image.split("/")[-2], result_image.split("/")[-1])
            line["addition"][0]
            attribute_1 = line["addition"][0]
            attribute_2 = line["addition"][1]
            # print(image_concept_path_1)
            # print(image_concept_path_2)
            # print(image_generated_path)
            concept_eva = f'''
            You will receive a generated image along with two knowledge concepts: {type_1} {concept_1} and {type_2} {concept_2}, and their corresponding attributes: {attribute_1} and {attribute_2}.
            Task:
            1. Confirmation of Presence: Check if both concepts are present in the image. If either concept is missing, the Task Score is 0.
            2. Attribute Verification: If both concepts are present, verify if {concept_1} possesses attribute {attribute_1} and {concept_2} possesses attribute {attribute_2}. If both conditions are met, the Task Score is 1; otherwise, the score is 0.
            Evaluation Order: Please first confirm the presence of the concepts, then proceed to verify their respective attributes.

            Output Format (Note: do not use any bold formatting):
            Task Score:
            Reason:
            '''               
            return concept_eva
        else:
            print("Wrong Task")
    elif bandf:
        if line["task"] == "action":
            type_1 = line["type"][0]
            type_2 = line["type"][1]
            concept_1 = line["concept"][0].replace("_", " ")
            concept_2 = line["concept"][1].replace("_", " ")
            action_1 = line["addition"][0]
            action_2 = line["addition"][1]
            concept_eva = f"""
            You are provided with a generated image, two knowledge concepts, {type_1} {concept_1} and {type_2} {concept_2}, along with two actions: {action_1} and {action_2}. Your task is as follows:
            Confirm Concept Presence: Check whether both {concept_1} and {concept_2} are present in the image. If either is not present, the Task Score should be 0.
            Confirm Actions: If both concepts are present, determine whether {type_1} {concept_1} is dynamically performing or being acted upon by the action of {action_1}, and {type_2} {concept_2} is dynamically performing or being acted upon by {action_2}. If both actions are completed, the Task Score should be 1; if either action is not completed, the Task Score should be 0.
            Evaluation Order: Please first confirm the presence of both concepts, then verify the actions for each knowledge concept.

            Output Format (Note: the output must be without any bold formatting):
            Task Score:
            Reason:
            """
            return concept_eva
        elif line["task"] == "background variation" or line["task"] == "variation":
            type_ = line["type"][2]
            concept = line["concept"][2].replace("_", " ")
            attribute = line["addition"][0]
            concept_eva = f'''
            You will be provided with a generated image, a background knowledge concept {type_} {concept}, and an attribute related to background: {attribute}. Your task is:
            1. Confirm Concept Presence: Check whether {concept} is present in the image. If it is not present, the Task Score should be 0.
            2. Confirm Background Attribute: If {concept} is present, assess whether the related background exhibits the attribute {attribute}. If it does, the Task Score should be 1; otherwise, the Task Score should be 0.
            Evaluation Order: Please first confirm the presence of the concept, then verify its background attribute.

            Output Format (Note: do not use any bold formatting):
            Task Score:
            Reason:
            '''    
            return concept_eva
        elif line["task"] == "interaction":
            type_1 = line["type"][0]
            type_2 = line["type"][1]
            concept_1 = line["concept"][0]
            concept_2 = line["concept"][1]
            concept_image_1 = line["reference_image"][0]
            concept_image_2 = line["reference_image"][1]

            task = "interaction"
            text = line["sentence"]
            
            # image_concept_path_1 = os.path.join(concept_base, concept_image_1.split("/")[-2], concept_image_1.split("/")[-1])
            # image_concept_path_2 = os.path.join(concept_base, concept_image_2.split("/")[-2], concept_image_2.split("/")[-1])
            # result_image = line["result_image"]
            # image_generated_path = os.path.join(generate_base, result_image.split("/")[-2], result_image.split("/")[-1])
            addition = line["addition"][0]
            # print(image_concept_path_1)
            # print(image_concept_path_2)
            # print(image_generated_path)
            concept_eva = f'''
            You are provided with a generated image and two knowledge concepts: {type_1} {concept_1} and {type_2} {concept_2}. Additionally, an interaction action is specified: {addition}. Your task is to :
            1. Confirmation of Presence: determine whether both {concept_1} and {concept_2} are present in the generated image. If either is not present, the Task Score should be 0. If both are present:
            2. Interaction Verification: determine whether they are dynamically interacting through the specified interaction of {addition}. If this interaction is occurring, the Task Score should be 1; if not, the Task Score should be 0.
            Evaluation Order: Please first confirm the presence of the concepts, then proceed to verify their interaction.
            Input:
            1. The generated image: <image>.
            2. Concepts: {type_1} {concept_1}, {type_2} {concept_2}, Interaction: {addition}
            Output Format (Note: the output must be without any bold formatting):
            Task Score: 
            Reason: 
            '''               
            return concept_eva
        elif line["task"] == "size":
            type_1 = line["type"][0]
            type_2 = line["type"][1]
            concept_1 = line["concept"][0]
            concept_2 = line["concept"][1]
            concept_image_1 = line["reference_image"][0]
            concept_image_2 = line["reference_image"][1]

            task = "interaction"
            text = line["sentence"]
            
            # image_concept_path_1 = os.path.join(concept_base, concept_image_1.split("/")[-2], concept_image_1.split("/")[-1])
            # image_concept_path_2 = os.path.join(concept_base, concept_image_2.split("/")[-2], concept_image_2.split("/")[-1])
            # result_image = line["result_image"]
            # image_generated_path = os.path.join(generate_base, result_image.split("/")[-2], result_image.split("/")[-1])
            addition = line["addition"][0]
            # print(image_concept_path_1)
            # print(image_concept_path_2)
            # print(image_generated_path)
            concept_eva = f'''
            You need to analyze a generated image that contains two knowledge concepts: {type_1} {concept_1} (for example, "Husky") and {type_2} {concept_2} (for example, "Egyptian Cat"). The task details are as follows:
            1. Existence Check: First, confirm whether both concepts are present in the generated image. If either concept is missing, the Task Score should be 0.
            2. Size Restoration: If both concepts are present, assess whether their size proportions in the image match their real-world counterparts.
            3. Evaluation Criteria: Estimate the relative volume, height, or recognizable size ratio of each concept in the image to ensure they reflect the real-world size differences. If {concept_1} is indeed larger than {concept_2} as expected, the Task Score should be 1; otherwise, it should be 0.
            Input Information:

            The generated image: <image>.
            Concepts: {type_1} {concept_1} , {type_2} {concept_2}
            Output Format (Note: the output must be without any bold formatting):
            Task Score:
            Reason:
            '''               
            return concept_eva
        elif line["task"] == "differentiating":
            type_1 = line["type"][0]
            type_2 = line["type"][1]
            concept_1 = line["concept"][0].replace("_", " ")
            concept_2 = line["concept"][1].replace("_", " ")
            concept_image_1 = line["reference_image"][0]
            concept_image_2 = line["reference_image"][1]

            task = "interaction"
            text = line["sentence"]
            
            # image_concept_path_1 = os.path.join(concept_base, concept_image_1.split("/")[-2], concept_image_1.split("/")[-1])
            # image_concept_path_2 = os.path.join(concept_base, concept_image_2.split("/")[-2], concept_image_2.split("/")[-1])
            # result_image = line["result_image"]
            # image_generated_path = os.path.join(generate_base, result_image.split("/")[-2], result_image.split("/")[-1])
            line["addition"][0]
            attribute_1 = line["addition"][0]
            attribute_2 = line["addition"][1]
            # print(image_concept_path_1)
            # print(image_concept_path_2)
            # print(image_generated_path)
            concept_eva = f'''
            You will receive a generated image along with two knowledge concepts: {type_1} {concept_1} and {type_2} {concept_2}, and their corresponding attributes: {attribute_1} and {attribute_2}.
            Task:
            1. Confirmation of Presence: Check if both concepts are present in the image. If either concept is missing, the Task Score is 0.
            2. Attribute Verification: If both concepts are present, verify if {concept_1} possesses attribute {attribute_1} and {concept_2} possesses attribute {attribute_2}. If both conditions are met, the Task Score is 1; otherwise, the score is 0.
            Evaluation Order: Please first confirm the presence of the concepts, then proceed to verify their respective attributes.

            Output Format (Note: do not use any bold formatting):
            Task Score:
            Reason:
            '''               
            return concept_eva  
        else:
            print("Wrong Task")

def integration_prompt_process(line):
    if len(line["concept"]) == 3:
        type_1 = line["type"][0]
        type_2 = line["type"][1]
        type_3 = line["type"][2]
        concept_1 = line["concept"][0].replace("_", " ")
        concept_2 = line["concept"][1].replace("_", " ")
        concept_3 = line["concept"][2].replace("_", " ")
        concept_image_1 = line["reference_image"][0]
        concept_image_2 = line["reference_image"][1]
        concept_image_3 = line["reference_image"][2]

        # task = "interaction"
        text = line["sentence"]
        # image_concept_path_1 = os.path.join(concept_base, concept_image_1.split("/")[-2], concept_image_1.split("/")[-1])
        # image_concept_path_2 = os.path.join(concept_base, concept_image_2.split("/")[-2], concept_image_2.split("/")[-1])
        # image_concept_path_3 = os.path.join(concept_base, concept_image_3.split("/")[-2], concept_image_3.split("/")[-1])
        # result_image = line["result_image"]
        # image_generated_path = os.path.join(generate_base, result_image.split("/")[-2], result_image.split("/")[-1])
        addition = line["addition"][0]
        # print(image_concept_path_1)
        # print(image_concept_path_2)
        # print(image_concept_path_3)
        # print(image_generated_path)
        integration_eval = f'''
        You are provided with a generated image, the text prompt, and three knowledge concepts: {type_1} {concept_1}, {type_2} {concept_2} and background {type_3} {concept_3}. Your task is to evaluate the generated image based on these knowledge concepts and the text prompt.
        1. Confirmation of Presence: determine whether {concept_1}, {concept_2} and {concept_3} are present in the generated image. If any of them is not present, the Total Rating should be 0. If all are present:
        2. Integration Evaluation:
        - Seamless Transition: Assess whether the boundaries between {concept_1}, {concept_2}, and background {concept_3} are smooth. 
        - Visual Completeness: Evaluate whether {concept_1}, {concept_2} and the background {concept_3} exhibit visual consistency without unnecessary additions, missing parts, or unnatural appearances.
        - Authenticity: Assess if the size and position of {concept_1}, {concept_2}, {concept_3} are realistic for the environment. For instance, a car should be much larger than a husky, and neither should appear in illogical positions, such as floating without support.
        - Prompt Following: Evaluate whether the image faithfully represents all major elements specified in the text prompt and ensures that specific details, such as colors and shapes, are accurately depicted.
        Evaluation Order: Please first confirm the presence of the concepts, then proceed to evaluate the integration.
        Evaluation Criteria: Evaluate based on the criteria above. Total Rate from 0 to 4 based on how many criteria are fully met:
        0: None met
        1: One met
        2: Two met
        3: Three met
        4: All met
        Provide a concise explanation for the rating. Keep your feedback specific and to the point.
        Input:
        Generated image: <image>
        Text prompt: "{text}"

        Output Format (Notice that output must be without any bold formatting):
        Total Rating: Your Rating
        Seamless Transition: 0/1 Reason:
        Visual Completeness: 0/1 Reason:
        Authenticity: 0/1 Reason:
        Prompt Following: 0/1 Reason:
        '''    
        return integration_eval
    elif len(line["concept"]) == 2:
        type_1 = line["type"][0]
        type_2 = line["type"][1]
        concept_1 = line["concept"][0].replace("_", " ")
        concept_2 = line["concept"][1].replace("_", " ")
        concept_image_1 = line["reference_image"][0]
        concept_image_2 = line["reference_image"][1]

        # task = "interaction"
        text = line["sentence"]
        # image_concept_path_1 = os.path.join(concept_base, concept_image_1.split("/")[-2], concept_image_1.split("/")[-1])
        # image_concept_path_2 = os.path.join(concept_base, concept_image_2.split("/")[-2], concept_image_2.split("/")[-1])
        # result_image = line["result_image"]
        # image_generated_path = os.path.join(generate_base, result_image.split("/")[-2], result_image.split("/")[-1])
        addition = line["addition"][0]
        # print(image_concept_path_1)
        # print(image_concept_path_2)
        # print(image_generated_path)
        integration_eval = f'''
        You are provided with a generated image, the text prompt, and two knowledge concepts: {type_1} {concept_1} and {type_2} {concept_2}. Your task is to evaluate the generated image based on these knowledge concepts and the text prompt.
        1. Confirmation of Presence: determine whether both {concept_1} and {concept_2} are present in the generated image. If either is not present, the Total Rating should be 0. If both are present:
        2. Integration Evaluation:
        - Seamless Transition: Assess whether the boundaries between {concept_1} and {concept_2} are smooth, and ensure they integrate harmoniously with the surrounding environment.
        - Visual Completeness: Evaluate whether {concept_1} and {concept_2} exhibit visual consistency without unnecessary additions, missing parts, or unnatural appearances.
        - Authenticity: Assess if the size and position of {concept_1} and {concept_2} are realistic for the environment. For instance, a car should be much larger than a husky, and neither should appear in illogical positions, such as floating without support.
        - Prompt Following: Evaluate whether the image faithfully represents all major elements specified in the text prompt and ensures that specific details, such as colors and shapes, are accurately depicted.
        Evaluation Order: Please first confirm the presence of the concepts, then proceed to evaluate the integration.
        Evaluation Criteria: Evaluate based on the criteria above. Total Rate from 0 to 4 based on how many criteria are fully met:
        0: None met
        1: One met
        2: Two met
        3: Three met
        4: All met
        Provide a concise explanation for the rating. Keep your feedback specific and to the point.
        Input:
        Generated image: <image>
        Text prompt: "{text}"

        Output Format (Notice that output must be without any bold formatting):
        Total Rating: Your Rating
        Seamless Transition: 0/1 Reason:
        Visual Completeness: 0/1 Reason:
        Authenticity: 0/1 Reason:
        Prompt Following: 0/1 Reason:
        '''           
        return integration_eval
    else:
        print("Wrong Num")  