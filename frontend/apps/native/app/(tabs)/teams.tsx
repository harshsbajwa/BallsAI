import React from "react";
import { View, Text, FlatList, TouchableOpacity } from "react-native";
import { useQuery } from "@tanstack/react-query";
import { useRouter } from "expo-router";
import { apiClient } from "@/lib/api-client";
import { Loading } from "@/components/Loading";
import { Team } from "@nba/api/types";

export default function TeamsScreen() {
  const router = useRouter();

  const { data: teamsData, isLoading, error } = useQuery({
    queryKey: ["teams"],
    queryFn: () => apiClient.getTeams(30),
  });

  const handleTeamPress = (team: Team) => {
    router.push(`/team/${team.team_id}`);
  };

  if (isLoading) {
    return (
      <View className="flex-1 bg-gray-50">
        <Loading text="Loading teams..." />
      </View>
    );
  }

  if (error) {
    return (
      <View className="flex-1 bg-gray-50 items-center justify-center">
        <Text className="text-red-500 text-lg">Error loading teams</Text>
        <Text className="text-gray-500 text-sm mt-2">Please try again</Text>
      </View>
    );
  }

  return (
    <View className="flex-1 bg-gray-50 pt-4">
      <FlatList
        data={teamsData?.items}
        keyExtractor={(item) => item.team_id.toString()}
        renderItem={({ item }) => (
          <TouchableOpacity
            onPress={() => handleTeamPress(item)}
            className="bg-white mx-4 my-2 rounded-lg p-4 shadow-sm"
          >
            <View className="flex-row justify-between items-center">
              <View className="flex-1">
                <Text className="font-bold text-lg text-gray-900">
                  {item.full_name}
                </Text>
                <Text className="text-gray-600 mt-1">{item.team_city}</Text>
              </View>
              <View className="bg-blue-100 px-3 py-1 rounded-full">
                <Text className="text-blue-800 font-bold">
                  {item.team_abbrev}
                </Text>
              </View>
            </View>
          </TouchableOpacity>
        )}
        contentInsetAdjustmentBehavior="automatic"
        showsVerticalScrollIndicator={false}
      />
    </View>
  );
}